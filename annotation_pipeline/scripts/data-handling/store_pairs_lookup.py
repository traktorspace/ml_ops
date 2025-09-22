# You can use this script to dump to a specific folder all the pairs
# LXX.tif and LXX_annotation.tif as unique plot for pair containing
# LXX.tif, LXX_annotation.tif and the overlay of the two
# Useful to detect misalignment or bad annotations

# python3 scripts/store_pairs_lookup.py.py
# -imgpath=/bigdata/datasets_analytics/clouds/hf1a_reflectance/pipeline_test
# -dumpdir=./media_test/
import matplotlib

matplotlib.use('Agg')
import argparse
import traceback
from pathlib import Path

from loguru import logger
from tqdm import tqdm

from annotation_pipeline.utils.path_utils import fetch_data_pair
from annotation_pipeline.utils.viz_utils import build_cube_and_mask_preview


def collect_prod_names(
    img_root: Path, all_dirs: bool, listfile: Path | None
) -> list[str]:
    """
    Return the list of sub-directory names to work on.
    """
    if all_dirs:
        return [p.name for p in img_root.iterdir() if p.is_dir()]

    # listfile mode
    if listfile is not None:
        with listfile.open() as fh:
            requested = [line.strip() for line in fh if line.strip()]
        missing = [name for name in requested if not (img_root / name).exists()]
        if missing:
            raise FileNotFoundError(
                f'The following products listed in {listfile} do not exist under {img_root}: '
                f'{", ".join(missing)}'
            )
    else:
        raise RuntimeError(
            'You need to specify one usage case by either setting all_dirs flag or specifying a listfile path'
        )
    return requested


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '-imgpath',
            type=Path,
            required=True,
            help='Path where your pairs of img+annotation are stored',
        )
        parser.add_argument(
            '-dumpdir',
            type=Path,
            required=True,
            help='Path where you want to store the plots of each cube+annotation pair',
        )

        args = parser.parse_args()

        if not args.imgpath.exists():
            raise FileNotFoundError('Img path not found')

        prod_names = [p.name for p in list(Path(args.imgpath).glob('*'))]
        logger.info(f'Found {len(prod_names)}')

        for prod_name in tqdm(prod_names):
            cube_path, annotation_path = fetch_data_pair(
                root_dir=Path(args.imgpath),
                prod_name=str(prod_name),
            )

            buf = build_cube_and_mask_preview(
                cube_path=cube_path,
                mask_path=annotation_path,
                overlay_alpha=0.3,
                dpi=300,
            )

            out_file = args.dumpdir / f'{prod_name}.png'
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_bytes(buf.getvalue())

        logger.info('Operation completed!')

    except Exception:
        logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
