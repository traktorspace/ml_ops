from pathlib import Path

from google.cloud import storage


def fetch_google_bucket(project: str, bucket_name: str):
    try:
        client = storage.Client(project)
        return client.bucket(bucket_name)
    except Exception as e:
        raise e


def upload_folder_to_bucket(
    folder_to_copy: str | Path,
    gcp_dst_folder: str,
    bucket: storage.bucket.Bucket,
):
    if isinstance(folder_to_copy, str):
        folder_to_copy = Path(folder_to_copy)

    folder_name = folder_to_copy.name
    folder_prefix = f'{gcp_dst_folder}/{folder_name}/'

    if any(bucket.list_blobs(prefix=folder_prefix, max_results=1)):
        print(
            f"Skip: '{folder_prefix}' already exists in bucket '{bucket.name}'."
        )
        return

    for file_path in folder_to_copy.rglob('*'):
        if not file_path.is_file():
            continue

        rel_path = file_path.relative_to(folder_to_copy).as_posix()
        blob_path = f'{folder_prefix}{rel_path}'
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(file_path)
        print(f'Uploaded {file_path} to {blob_path}')


def list_blob_folders_at_depth(
    bucket: storage.bucket.Bucket, depth: int
) -> list:
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
