import io
from dataclasses import dataclass, field
from pathlib import Path
from shutil import rmtree

import numpy as np
import psycopg
import rasterio
from encord.project import Project
from frame_aligner.methods import BaseAligner
from loguru import logger

from data_pipeline.utils.cube_utils import (
    align_label_with_new_reference,
    apply_nan_mask_to_annotation,
    get_aligner,
    store_annotation,
)
from data_pipeline.utils.db_utils import exec_query
from data_pipeline.utils.encord_utils import get_annotation_tensor
from data_pipeline.utils.path_utils import (
    ensure_dir,
    infer_product_level,
    make_symlink,
)


@dataclass(slots=True)
class JobStats:
    annotation_not_found: list[str] = field(default_factory=list)
    bad_shape_found: list[str] = field(default_factory=list)
    processing_failed: list[str] = field(default_factory=list)
    correctly_processed: list[str] = field(default_factory=list)
    skipped_existing: list[str] = field(default_factory=list)

    def _bad_shape_unprocessed(self) -> set[str]:
        """
        Return the subset of `bad_shape_found` that never made it into
        `correctly_processed`.
        """
        return set(self.bad_shape_found) - set(self.correctly_processed)

    def add(self, bucket: str, product: str) -> None:
        """Append the prod-name to the bucket identified by *bucket*."""
        getattr(self, bucket).append(product)

    def summary(self) -> dict[str, int]:
        """Return a {bucket → count} dict (convenience for printing)."""
        return {name: len(getattr(self, name)) for name in self.__annotations__}

    def as_markdown(self) -> str:
        """
        Return all collected product names as **one** fenced-code block.

        Example
        -------
        >>> report = job_processor.as_markdown()
        >>> print(report)
        ```
        prod1
        prod2
        ```

        If no products are present, an empty string is returned.
        """
        lines: list[str] = []

        for bucket in self.__annotations__:  # keep declaration order
            items: list[str] = getattr(self, bucket)
            if not items:  # skip empty categories
                continue

            lines.append(f'*{bucket}* ({len(items)})')
            lines.append('```')
            lines.extend(items)
            lines.append('```')
            lines.append('')  # blank line between sections

        # Extra constraint to check files with bad shape that
        # haven't been correclty processed
        missing = self._bad_shape_unprocessed()
        if missing:
            lines.append(f'*bad_shape_unprocessed* ({len(missing)})')
            lines.append('```')
            lines.extend(sorted(missing))
            lines.append('```')
            lines.append('')

        return '\n'.join(lines).rstrip()

    def as_text_buffers(self) -> dict[str, io.BytesIO]:
        """
        Return one BytesIO per non-empty bucket.
        The buffer's .name is set so libraries such as slack-sdk
        use it as the file name.

        Example returned value
        ----------------------
        {
            'annotation_not_found.txt': <BytesIO>,
            'correctly_processed.txt': <BytesIO>,
            ...
        }
        """
        buffers: dict[str, io.BytesIO] = {}

        def _make_buffer(name: str, lines: list[str]) -> io.BytesIO:
            # trailing '\n' so final line gets its own newline in viewers
            raw = ('\n'.join(lines) + '\n').encode('utf-8')
            buf = io.BytesIO(raw)
            buf.seek(0)
            buf.name = f'{name}.txt'  # slack_sdk looks at .name
            return buf

        # regular buckets
        for bucket in self.__annotations__:  # keep declaration order
            items: list[str] = getattr(self, bucket)
            if items:  # skip empty ones
                buffers[bucket] = _make_buffer(bucket, sorted(items))

        # extra check
        missing = self._bad_shape_unprocessed()
        if missing:
            buffers['bad_shape_unprocessed'] = _make_buffer(
                'bad_shape_unprocessed', sorted(missing)
            )

        return buffers


@dataclass(slots=True)
class JobProcessor:
    dst: Path
    db_conn: psycopg.Connection
    q_success: str  # SQL → mark success
    q_failure: str  # SQL → write exception
    encord_project: Project  # the Encord project handle
    aligners: list[BaseAligner] = None  # lazy-initialised
    allow_overwriting: bool = False
    stats: JobStats = field(default_factory=JobStats)
    dry_run: bool = False

    def _align_annotation_if_needed(
        self,
        label: np.ndarray,
        job: dict[str, str],
        ref_shape: tuple[int, int | None],
    ) -> tuple[np.ndarray, dict]:
        if label.shape == ref_shape:
            return label, None  # nothing to do

        logger.info(
            f'Bad shape found for {job["prodname"]} - running alignment'
        )
        self.stats.add('bad_shape_found', job['prodname'])
        self.aligners = self.aligners or get_aligner()  # init once

        new_label, new_prof = align_label_with_new_reference(
            old_frame_path=job['remote_annotation_path'],
            new_frame_path=job['s3_path'],
            aligners=self.aligners,
            old_label=label,
        )

        if new_label.shape != ref_shape:
            logger.error(f'Processing failed for {job["prodname"]}')
            self.stats.add('processing_failed', job['prodname'])
            raise RuntimeError(f'Alignment failed for {job["prodname"]}')

        logger.success(f'Alignment was succesfull for {job["prodname"]}')
        return new_label, new_prof

    def __call__(self, db_id: int, job: dict[str, str]) -> None:
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
            # register a fake “success” so that statistics still make sense
            self.stats.mark_success(db_id)
            return

        logger.info(f'Starting job {db_id} (prod={job["prodname"]})')
        dst = self.dst / job['prodname']

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
        with rasterio.open(job['s3_path']) as src:
            ref_data = src.read(1)
            ref_shape = ref_data.shape
            ref_profile = src.profile | {'count': 1}

        # 3) Align annotation to cube if needed
        label, profile = self._align_annotation_if_needed(label, job, ref_shape)
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
        make_symlink(
            destination=dst / f'{prod_level}.tif',
            path_to_link=job['s3_path'],
        )

        # 5) Update db entry
        exec_query(
            self.db_conn, self.q_success, params={'annotation_id': db_id}
        )
        self.stats.add('correctly_processed', job['prodname'])
