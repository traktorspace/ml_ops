# python3 scripts/pairs_lookup.py
# -imgpath=/bigdata/datasets_analytics/clouds/hf1a_reflectance/pipeline_test
# -dumpdir=./media_test/
import matplotlib

matplotlib.use('Agg')
import argparse
import traceback
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from data_pipeline.utils.path_utils import fetch_data_pair
from data_pipeline.utils.viz_utils import build_cube_annotation_preview


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-imgpath', type=Path, required=True)
        parser.add_argument('-dumpdir', type=Path, required=True)

        args = parser.parse_args()

        if not args.imgpath.exists():
            raise FileNotFoundError('Img path not found')

        prod_names = [p.name for p in list(Path(args.imgpath).glob('*'))]
        logger.info(f'Found {len(prod_names)}')

        for prod_name in tqdm(prod_names):
            cube_path, annotation_path = fetch_data_pair(
                root_dir=args.imgpath,
                prod_name=prod_name,
            )

            buf = build_cube_annotation_preview(
                cube_path=cube_path,
                annotation_path=annotation_path,
                overlay_alpha=0.3,
                dpi=150,
            )

            out_file = args.dumpdir / f'{prod_name}.png'
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(buf.getvalue())

        logger.info('Operation completed!')

    except Exception:
        logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
