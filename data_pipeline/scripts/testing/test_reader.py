import os
from pathlib import Path

from dotenv import load_dotenv
from fs.memoryfs import MemoryFS

from data_pipeline.utils.gcloud_utils import (
    BucketWrapper,
)
from data_pipeline.utils.path_utils import unzip_any


def main():
    mem = MemoryFS()
    #
    # /bigdata/hyperfield/processed/2025/08/25/hyperfield-1a/hyperfield1a_L1B_20250825T024340/
    # f1, f2, outdir = extract_bands(
    #     filepath=Path(
    #         '/bigdata/hyperfield/processed/2025/08/06/hyperfield-1a/hyperfield1a_L1B_20250806T035000/L1B.tif'
    #     ),
    #     save_product=True,
    #     dst_path=Path('temp_fs'),
    #     memory_filesystem=mem,
    #     verbose=True,
    #     stretch_contrast=False,
    # )

    # with open('f1.png', 'wb') as f:
    #     f.write(f1.read() if hasattr(f1, "read") else f1)
    load_dotenv(
        '/home/mlops/repos/ml_ops/data_pipeline/configs/cloud_annotation/.env'
    )
    # print(outdir)
    # # for path, dirs, files in mem.walk("/", search="depth"):
    # #     print("Directory:", path)
    # #     print("  Subdirs:", dirs)
    # #     print("  Files:", files)

    # # with mem.open('temp_fs/hyperfield1a_L1C_20250717T095347/hyperfield1a_L1C_20250717T095347_rgb.png', 'rb') as f:
    # #     print(f.seek(0))

    bw = BucketWrapper(
        project_name=os.environ['DATA_ARCHIVE_GOOGLE_CLOUD_PROJECT'],
        bucket_name=os.environ['DATA_ARCHIVE_GOOGLE_CLOUD_BUCKET_NAME'],
    )
    print(bw.project_name)
    print(bw.bucket_name)
    bw.instantiate_bucket()
    # bw.download_product(
    #     blob_name='2025/06/01/hyperfield-1a/hyperfield1a_L1B_20250601T101412.zip',
    #     dst='./tmp_download',
    # )
    blobpath = Path(
        '2025/06/01/hyperfield-1a/hyperfield1a_L1B_20250601T101412.zip'
    )
    ppp = bw.download_product(
        blob_name=blobpath.as_posix(),
        dst=mem,
        memfs_path=f'./downloads/{blobpath.name}',
        verbose=True,
    )
    # bw.th =
    print('path ->', ppp)
    print()

    print(mem.tree())
    unzip_any(
        src=ppp,
        dst=Path(ppp).as_posix().replace('.zip', ''),
        src_fs=mem,
        dst_fs=mem,
    )
    # upload_folder_to_bucket(
    #     src=mem,
    #     gcp_dst_folder='cloud_annotations',
    #     bucket=bucket,
    #     memfs_subdir=str(outdir),
    # )
    # mem.close()
    # del mem
    print(mem.tree())


if __name__ == '__main__':
    main()
