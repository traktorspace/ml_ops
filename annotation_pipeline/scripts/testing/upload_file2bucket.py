"""
upload_file2bucket.py

Upload a single local file (e.g. a ZIP) to a Google-Cloud-Storage bucket.

Example
-------
python upload_zip2bucket.py \
    --project my-gcp-project \
    --bucket  my-data-bucket \
    --src     ./data/archive.zip \
    --dst     datasets/raw            # (uploaded as datasets/raw/archive.zip)
"""

import argparse
from pathlib import Path

from loguru import logger

from annotation_pipeline.utils.gcloud_utils import (
    BucketWrapper,
    upload_file_to_bucket,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description='Upload a single file to a GCS bucket.'
    )
    p.add_argument(
        '--project',
        required=True,
        help='GCP project ID that owns the bucket.',
    )
    p.add_argument(
        '--bucket',
        required=True,
        help='Destination bucket name.',
    )
    p.add_argument(
        '--src',
        required=True,
        type=Path,
        help='Path to the local file you want to upload.',
    )
    p.add_argument(
        '--dst',
        required=True,
        help=(
            'Destination *prefix* (folder) inside the bucket. '
            'The uploaded object will be '
            '`<dst>/<src_file_name>`.'
        ),
    )
    p.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite if an object with the same name already exists.',
    )
    return p.parse_args()


def main() -> None:
    try:
        args = parse_args()

        bw = BucketWrapper(project_name=args.project, bucket_name=args.bucket)
        bw.instantiate_bucket()

        blob_name = upload_file_to_bucket(
            src=args.src,
            gcp_dst_folder=args.dst,
            bucket=bw.bucket_instance,  # type: ignore[arg-type]
            overwrite=args.overwrite,
        )

        logger.success(f'Done. Object stored as gs://{args.bucket}/{blob_name}')
    except Exception as e:
        logger.error(f'Error occured: {e}')


if __name__ == '__main__':
    main()
