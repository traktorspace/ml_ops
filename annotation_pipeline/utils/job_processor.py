import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from shutil import rmtree

import numpy as np
import psycopg
import rasterio
from encord.project import Project
from frame_aligner.methods import BaseAligner
from fs.memoryfs import MemoryFS
from google.cloud.storage.bucket import Bucket
from loguru import logger
from rasterio.io import MemoryFile

from annotation_pipeline.utils.cube_utils import (
    align_label_with_new_reference,
    apply_nan_mask_to_annotation,
    extract_bands,
    get_aligner,
    store_annotation,
)
from annotation_pipeline.utils.db_utils import exec_query
from annotation_pipeline.utils.encord_utils import (
    ImageGroup,
    ImageGroupPayload,
    get_annotation_tensor,
)
from annotation_pipeline.utils.gcloud_utils import (
    BucketWrapper,
    upload_folder_to_bucket,
)
from annotation_pipeline.utils.path_utils import (
    ensure_dir,
    infer_product_level,
    make_symlink,
    unzip_any,
)


@dataclass(slots=True)
class JobStats:
    """
    Base-class that takes care of all generic bookkeeping logic.
    Sub-classes only have to

      1. declare their own buckets (i.e. dataclass fields)
      2. optionally override _extra_sections() to expose
         additional, computed sections.
    """

    # -------------  public helpers ------------- #
    def add(self, bucket: str, product: str) -> None:  # noqa: D401
        """Append *product* to *bucket* (e.g. ``stats.add('failed', 'p1')``)."""
        getattr(self, bucket).append(product)

    def summary(self) -> dict[str, int]:
        """{bucket → #items} - convenience for logging/printing."""
        return {name: len(getattr(self, name)) for name in self.__annotations__}

    def _extra_sections(self) -> dict[str, list[str]]:
        """
        Hook for subclasses to return additional *virtual* buckets.

        Must return a mapping bucket-name → list[str] (even if empty).
        """
        return {}

    def as_markdown(self) -> str:
        """
        One fenced-code block per non-empty bucket.

        ```
        *failed* (2)
        ```
        p1
        p2
        ```
        """
        lines: list[str] = []

        # real buckets declared on the dataclass --------------------------
        for bucket in self.__annotations__:  # keep declaration order
            items: list[str] = getattr(self, bucket)
            if not items:
                continue

            lines.append(f'*{bucket}* ({len(items)})')
            lines.append('```')
            lines.extend(items)
            lines.append('```')
            lines.append('')  # blank line

        # virtual / computed buckets -------------------------------------
        for bucket, items in self._extra_sections().items():
            if not items:
                continue
            lines.append(f'*{bucket}* ({len(items)})')
            lines.append('```')
            lines.extend(sorted(items))
            lines.append('```')
            lines.append('')

        return '\n'.join(lines).rstrip()

    # ───────────────────────────────────────────────────────────────────── #
    def as_text_buffers(self) -> dict[str, io.BytesIO]:
        """
        One in-memory file per non-empty bucket.
        The buffer's .name is set so that e.g. slack-sdk uses it as filename.
        """

        def _make_buffer(name: str, lines: list[str]) -> io.BytesIO:
            raw = ('\n'.join(lines) + '\n').encode('utf-8')  # newline at EOF
            buf = io.BytesIO(raw)
            buf.name = f'{name}.txt'
            buf.seek(0)
            return buf

        buffers: dict[str, io.BytesIO] = {}

        # real buckets
        for bucket in self.__annotations__:
            items: list[str] = getattr(self, bucket)
            if items:
                buffers[bucket] = _make_buffer(bucket, sorted(items))

        # virtual / computed buckets
        for bucket, items in self._extra_sections().items():
            if items:
                buffers[bucket] = _make_buffer(bucket, sorted(items))

        return buffers


@dataclass(slots=True)
class JobPullStats(JobStats):
    annotation_not_found: list[str] = field(default_factory=list)
    bad_shape_found: list[str] = field(default_factory=list)
    processing_failed: list[str] = field(default_factory=list)
    correctly_processed: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)

    # ---------- extra, pull-specific helpers ---------- #
    def _bad_shape_unprocessed(self) -> list[str]:
        """bad_shape files that *never* ended up in correctly_processed."""
        return list(set(self.bad_shape_found) - set(self.correctly_processed))

    # ---------- override virtual bucket hook ---------- #
    def _extra_sections(self) -> dict[str, list[str]]:  # noqa: D401
        missing = self._bad_shape_unprocessed()
        return {'bad_shape_unprocessed': missing} if missing else {}


@dataclass(slots=True)
class JobPushStats(JobStats):
    """
    Example pull-workflow buckets.
    """

    skipped_existing: list[str] = field(default_factory=list)
    correctly_uploaded: list[str] = field(default_factory=list)
    not_uploaded: list[str] = field(default_factory=list)


@dataclass(slots=True)
class JobPullProcessor:
    dst: Path
    db_conn: psycopg.Connection
    q_success: str  # SQL → mark success
    q_failure: str  # SQL → write exception
    encord_project: Project  # the Encord project handle
    aligners: list[BaseAligner] | None = None  # lazy-initialised
    allow_overwriting: bool = False
    use_cloud_data_source: bool = False
    cloud_bucket_wrapper: BucketWrapper | None = None
    memfs = MemoryFS()
    stats: JobPullStats = field(default_factory=JobPullStats)
    save_only_annotation: bool = False
    aligner_method: str = 'roma'
    aligner_device: str = 'cuda:1'
    dry_run: bool = False

    def set_cloud_bucket_wrapper(self, cloud_bucket_wrapper: BucketWrapper):
        self.cloud_bucket_wrapper = cloud_bucket_wrapper

    def _align_annotation_if_needed(
        self,
        label: np.ndarray,
        job: dict[str, str],
        ref_shape: tuple[int, int | None],
        new_frame_src: str | Path | bytes,
    ) -> tuple[np.ndarray, dict | None]:
        """
        Verify that *label* matches the expected reference shape and, when it
        does not, realign it against the latest reference frame.

        The reference frame can be supplied as
        1. a local/remote file path (`str` or `Path`), or
        2. raw TIFF bytes (e.g. when the file lives in a ``MemoryFS``).

        Parameters
        ----------
        label : np.ndarray
            Annotation mask that may need re-alignment.
        job : dict[str, str]
            Metadata for the current job. Must contain at least
            ``'prodname'``, ``'remote_annotation_path'`` and ``'s3_path'``.
        ref_shape : tuple[int, int | None]
            Expected shape *(rows, cols)* of the annotation mask.
        new_frame_src : str | Path | bytes
            Source of the NEW reference frame that the annotation has to
            align to. Accepts a filesystem path or an in-memory byte string.

        Returns
        -------
        tuple[np.ndarray, dict | None]
            new_label :
                Annotation tensor whose shape now equals *ref_shape*.
            new_prof :
                Raster profile returned by the aligner, or ``None`` when no
                re-alignment was required.

        Raises
        ------
        RuntimeError
            Raised if re-alignment is attempted but the resulting label still
            does not match *ref_shape*.
        """
        if label.shape == ref_shape:
            return label, None  # nothing to do

        logger.info(
            f'Bad shape found for {job["prodname"]} - running alignment'
        )
        self.stats.add('bad_shape_found', job['prodname'])
        self.aligners = self.aligners or get_aligner(
            aligner_name=self.aligner_method,
            aligner_parameters={'device': self.aligner_device},
        )  # init once

        new_label, new_prof = align_label_with_new_reference(
            old_frame_src=Path(job['remote_annotation_path']),
            new_frame_src=new_frame_src,
            aligners=self.aligners,
            old_label=label,
        )

        if new_label.shape != ref_shape:
            logger.error(f'Processing failed for {job["prodname"]}')
            self.stats.add('processing_failed', job['prodname'])
            raise RuntimeError(f'Alignment failed for {job["prodname"]}')

        logger.success(f'Alignment was succesfull for {job["prodname"]}')
        return new_label, new_prof

    def __call__(self, db_id: str, job: dict[str, str]) -> None:
        """
        Process a single job.
        On success: commits artifacts + runs *q_success*.
        On any exception: let caller handle logging / q_failure.
        """
        if self.dry_run:
            logger.info(
                f'[DRY-RUN] would create symlink {job["s3_path"]} '
                f'and download annotation {job["remote_annotation_path"]}'
            )
            # TODO: register fake success?
            return

        logger.info(f'Starting job {db_id} (prod={job["prodname"]})')
        dst = self.dst / job['prodname']
        memfs_unzip_filepath = None
        memfs_raw_download_path = None
        # Skip completely if folder already exists and we do NOT overwrite
        if dst.exists() and not self.allow_overwriting:
            logger.info(
                f'Skipping job {db_id} (prod={job["prodname"]}) because of exists and override is disabled!'
            )
            self.stats.add('skipped_existing', job['prodname'])
            return

        # 1) Fetch annotation
        label = get_annotation_tensor(job['prodname'], self.encord_project)
        if label is None:
            logger.error(f'Annotation not found for {job["prodname"]}')
            self.stats.add('annotation_not_found', job['prodname'])
            raise LookupError(f'Annotation missing for {job["prodname"]}')

        # 2) Open reference raster from where it belongs (batman)
        if not self.use_cloud_data_source:
            with rasterio.open(job['s3_path']) as src:
                ref_data = src.read(1)
                ref_shape = ref_data.shape
                ref_profile = src.profile | {'count': 1}
                new_frame_src = Path(job['s3_path'])

        # Need to download files from remote bucket
        else:
            if self.cloud_bucket_wrapper is None:
                raise ValueError(
                    'You need to instantiate a BucketWrapper first with `set_cloud_bucket_wrapper`!'
                )
            else:
                if self.cloud_bucket_wrapper.bucket_instance is None:
                    self.cloud_bucket_wrapper.instantiate_bucket()

                # Download the product.zip in the destination
                memfs_raw_download_path = (
                    f'./downloads/{Path(job["s3_path"]).name}'
                )
                memfs_zip_path = self.cloud_bucket_wrapper.download_product(
                    blob_name=job['s3_path'],
                    dst=self.memfs,
                    memfs_path=memfs_raw_download_path,
                    verbose=True,
                )
                memfs_unzip_dir = (
                    Path(memfs_zip_path).as_posix().replace('.zip', '')
                )
                unzip_any(
                    src=memfs_zip_path,
                    dst=memfs_unzip_dir,
                    src_fs=self.memfs,
                    dst_fs=self.memfs,
                )
                logger.success(
                    f'Blob succesfully unzipped at destination: {memfs_unzip_dir}'
                )
                memfs_unzip_filepath = f'{Path(memfs_unzip_dir) / str(infer_product_level(job["prodname"]))}.tif'
                with self.memfs.openbin(memfs_unzip_filepath) as fp:
                    new_frames_bytes = fp.read()
                    # feed bytes to GDAL through Rasterio’s MemoryFile
                    with MemoryFile(new_frames_bytes) as memfile:
                        with memfile.open() as src:
                            ref_data = src.read(1)
                            ref_shape = ref_data.shape
                            ref_profile = src.profile | {'count': 1}
                    new_frame_src = new_frames_bytes

        # 3) Align annotation to cube if needed
        label, profile = self._align_annotation_if_needed(
            label=label,
            job=job,
            ref_shape=ref_shape,
            new_frame_src=new_frame_src,
        )
        profile = profile or ref_profile

        # 3.1) Apply NaN mask to the annotation
        label = apply_nan_mask_to_annotation(
            annotation=label, src_raster=ref_data, fill_value=0
        )

        # 4) clean old artefacts if we DO overwrite and perform writing operation
        if dst.exists() and self.allow_overwriting:
            rmtree(dst)
        dst = ensure_dir(dst)
        prod_level = infer_product_level(job['prodname'])
        store_annotation(label, profile, dst / f'{prod_level}_annotation.tif')
        # Products are in the same disk so we can just create a symlink
        if not self.use_cloud_data_source:
            if not self.save_only_annotation:
                make_symlink(
                    destination=dst / f'{prod_level}.tif',
                    path_to_link=job['s3_path'],
                )
                logger.success(
                    f'Succesfullycreated symlink for product {job["prodname"]}'
                )
        # Download from the RAM to file the file
        else:
            if not self.save_only_annotation:
                with open(dst / f'{prod_level}.tif', 'wb') as wf:
                    self.memfs.download(str(memfs_unzip_filepath), wf)
                logger.success(
                    f'Succesfully wrote from RAM to Disk the product {job["prodname"]}'
                )
            self.memfs.removetree('./downloads/')
        # 5) Update db entry
        exec_query(
            self.db_conn, self.q_success, params={'annotation_id': db_id}
        )
        self.stats.add('correctly_processed', job['prodname'])

    def __del__(self) -> None:
        try:
            if not self.memfs.isclosed():
                self.memfs.close()
        except Exception:
            # Never allow exceptions to propagate out of __del__
            pass


@dataclass(slots=True)
class JobPushProcessor:
    dst_bucket: Bucket
    bucket_root_folder: str
    dst_bucket_folder: str
    db_conn: psycopg.Connection
    q_success: str  # SQL → mark success
    q_failure: str  # SQL → write exception
    allow_overwriting: bool = False
    stats: JobPushStats = field(default_factory=JobPushStats)
    dry_run: bool = False
    memfs = MemoryFS()
    encord_payload: ImageGroupPayload = field(default_factory=ImageGroupPayload)
    encord_payload_dst: str | Path = 'encord_payload.json'

    def __call__(self, db_id: str, prod_path: str) -> None:
        # Save in a membuffer a folder containing the two pngs (rgb and nir)
        if self.dry_run:
            logger.info(f'[DRY-RUN] would work on {prod_path}')

        INTERNAL_DST_PATH = 'temp_fs'
        prodname = Path(prod_path).name
        PROD_LEVEL = infer_product_level(prodname)
        prodfullpath = Path(prod_path) / f'{PROD_LEVEL}.tif'
        _, _, outdir, rgb_path, nir_path = extract_bands(
            filepath=prodfullpath,
            save_product=True,
            dst_path=Path(INTERNAL_DST_PATH),
            memory_filesystem=self.memfs,
        )

        # Upload to google cloud
        uploaded_bucket_path = None
        if not self.dry_run:
            try:
                uploaded_bucket_path = upload_folder_to_bucket(
                    src=self.memfs,
                    gcp_dst_folder=self.dst_bucket_folder,
                    bucket=self.dst_bucket,
                    memfs_subdir=str(outdir),
                )
            except Exception:
                self.stats.add('not_uploaded', prodname)

            if uploaded_bucket_path is None:
                self.stats.add('skipped_existing', prodname)
                self.stats.add('not_uploaded', prodname)
                return

            else:
                logger.success(f'Correctly uploaded: {prodname}')
                self.stats.add('correctly_uploaded', prodname)
        else:
            logger.info('[DRY-RUN] files would be uploaded to remote bucket...')
            self.stats.add('not_uploaded', prodname)

        # Add entry to encord payload
        bucket_base = f'gs://{self.bucket_root_folder}'  # → 'gs://my-bucket'

        nir_key = (
            Path(self.dst_bucket_folder) / prodname / Path(str(nir_path)).name
        )
        rgb_key = (
            Path(self.dst_bucket_folder) / prodname / Path(str(rgb_path)).name
        )

        self.encord_payload.add_group(
            ImageGroup(
                title=prodname,
                createVideo=False,
                objectUrl_0=f'{bucket_base}/{nir_key.as_posix()}',
                objectUrl_1=f'{bucket_base}/{rgb_key.as_posix()}',
            )
        )

        # Update db
        if not self.dry_run and uploaded_bucket_path:
            exec_query(
                connection=self.db_conn,
                query=self.q_success,
                params=(
                    str(prodfullpath.parent.name),
                    datetime.now(timezone.utc),
                    uploaded_bucket_path,
                    True,
                    False,
                    None,
                    db_id,
                ),
            )
        else:
            logger.warning("Can't fill the db entry.")

        logger.info(f'Clearing up {outdir} MemFS')
        self.memfs.removetree(str(outdir))

    def finalize(self):
        self.encord_payload.to_json_file(path=self.encord_payload_dst)
        logger.success(f'Encord payload saved at {self.encord_payload_dst}')

    def __del__(self) -> None:
        try:
            if not self.memfs.isclosed():
                self.memfs.close()
        except Exception:
            # Never allow exceptions to propagate out of __del__
            pass
