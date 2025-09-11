import time
from dataclasses import dataclass
from pathlib import Path

from fs import path as fspath
from fs.base import FS
from fs.walk import Walker
from google.cloud import storage
from google.cloud.storage.bucket import Bucket
from loguru import logger

_MIB = 1_048_576


def fetch_google_bucket(project: str, bucket_name: str) -> Bucket:
    """Returns a Google Bucket instance given
    the project name and the bucket name"""
    try:
        client = storage.Client(project)
        return client.bucket(bucket_name)
    except Exception as e:
        raise e


def upload_folder_to_bucket(
    src: str | Path | FS,
    gcp_dst_folder: str,
    bucket: Bucket,
    memfs_subdir: str = '.',
) -> str | None:
    """
    Upload a local directory or a PyFilesystem (`MemoryFS`)
    subtree to a Google Cloud Storage bucket.

    Parameters
    ----------
    src : str | Path | fs.base.FS
        Source folder.
        - `str` / `Path` → local on-disk directory.
        - `FS` → any PyFilesystem object; only the contents of
          `memfs_subdir` are uploaded.
    gcp_dst_folder : str
        Destination “folder” (prefix) in the bucket, e.g. `"datasets/raw"`.
    bucket : google.cloud.storage.bucket.Bucket
        Target GCS bucket instance.
    memfs_subdir : str, default "."
        Relative path inside `src` (when it is an `FS`) to copy.
        `"."` copies the whole filesystem.

    Returns
    -------
    str | None
        The final GCS prefix that was created, or `None` if the prefix
        already existed and the upload was skipped.

    Raises
    ------
    TypeError
        If `src` is not `str`, `Path`, or `fs.base.FS`.
    """

    # Local disk behavior
    if isinstance(src, (str, Path)):
        root = Path(src)
        files = (p for p in root.rglob('*') if p.is_file())

        def openf(p):
            return open(p, 'rb')

        def rel(p):
            return p.relative_to(root).as_posix()

        folder = root.name

    # MemoryFS behavior
    elif isinstance(src, FS):
        files = Walker().files(src, path=memfs_subdir)

        def openf(p):
            return src.openbin(str(p))

        def rel(p):
            return Path(p).relative_to(memfs_subdir).as_posix()

        folder = folder = (
            Path(memfs_subdir).name
            if memfs_subdir not in ('.', '/', '')
            else 'memfs'
        )
    else:
        raise TypeError('src must be str | Path | MemoryFS')

    prefix = f'{gcp_dst_folder.rstrip("/")}/{folder}/'
    if any(bucket.list_blobs(prefix=prefix, max_results=1)):
        logger.warning(
            f"Skip: '{prefix}' already exists in bucket '{bucket.name}'."
        )
        return None  # already uploaded

    for f in files:
        blob = bucket.blob(prefix + rel(f))
        with openf(f) as fp:
            logger.info(f'Uploading {f}')
            blob.upload_from_file(fp)
    return prefix


def download_blob_from_bucket(  # noqa: N802  (keep original name)
    bucket,
    blob_name: str,
    dst: str | Path | FS,
    *,
    memfs_path: str | None = None,
    overwrite: bool = False,
    verbose: bool = False,
) -> str:
    """
    Download *blob_name* from *bucket* into `dst`.
    `dst` can be a path on disk or any PyFilesystem instance.
    """
    blob = bucket.get_blob(blob_name)
    if blob is None:
        raise FileNotFoundError(f'gs://{bucket.name}/{blob_name} not found')

    t0 = time.perf_counter()

    # ---------------------------------------------------------------- Disk ---
    if isinstance(dst, (str, Path)):
        p = Path(dst).expanduser().resolve()
        if p.is_dir() or (not p.exists() and p.suffix == ''):
            p /= Path(blob_name).name  # treat dst as directory
        if p.exists() and not overwrite:
            logger.warning(f"Skip: '{p}' already exists (overwrite=False).")
            return str(p)

        p.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Downloading gs://{bucket.name}/{blob_name} → {p}')
        blob.download_to_filename(p)
        final = p

    # --------------------------------------------------------- PyFS / memfs ---
    elif isinstance(dst, FS):
        fs_path = memfs_path or Path(blob_name).name

        parent = fspath.dirname(fs_path)
        if parent:
            dst.makedirs(parent, recreate=True)

        logger.info(
            f'Downloading gs://{bucket.name}/{blob_name} → memfs://{fs_path}'
        )
        with dst.openbin(fs_path, 'wb') as fh:
            blob.download_to_file(fh)
        final = fs_path

    else:
        raise TypeError('dst must be str | pathlib.Path | fs.base.FS')

    if verbose:
        logger.success(
            f'Downloaded {blob.size / _MIB:.2f} MiB in '
            f'{time.perf_counter() - t0:.2f}s'
        )

    return str(final)


def list_blob_folders_at_depth(bucket: Bucket, depth: int) -> list:
    """
    Returns the list of folder prefixes at exactly `depth` levels:
      depth=0 -> [""]         (the root)
      depth=1 -> ["a/", "b/", ...]
      depth=2 -> ["a/x/", "a/y/", "b/z/", ...]
    """
    # Start with root prexif
    prefixes = ['']
    for _ in range(depth):
        next_level = []
        for p in prefixes:
            # Fetch page iterator
            page = bucket.list_blobs(prefix=p, delimiter='/').pages
            for subpage in page:
                # subpage.prefixes is a set of next-level folders
                next_level.extend(subpage.prefixes)
        prefixes = next_level
    return prefixes


@dataclass(slots=True)
class BucketWrapper:
    project_name: str = ''
    bucket_name: str = ''
    bucket_instance: Bucket | None = None

    def instantiate_bucket(self):
        if self.bucket_instance is None:
            self.bucket_instance = fetch_google_bucket(
                project=self.project_name, bucket_name=self.bucket_name
            )
            logger.success('Bucket has been succesfully fetched.')
        else:
            logger.warning(
                'Bucket exists already, cannot instantiate a new one.'
            )

    def download_product(
        self,
        blob_name: str,
        dst: str | Path | FS,
        *,
        memfs_path: str | None = None,
        overwrite: bool = False,
        verbose: bool = False,
    ) -> str:
        if self.bucket_instance is None:
            raise RuntimeError('BucketWrapper.bucket_instance is None')
        return download_blob_from_bucket(
            self.bucket_instance,
            blob_name,
            dst,
            memfs_path=memfs_path,
            overwrite=overwrite,
            verbose=verbose,
        )
