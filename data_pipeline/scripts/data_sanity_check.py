# python3 scripts/data_sanity_check.py
#  -imgpath=/bigdata/datasets_analytics/clouds/hf1a_reflectance/pipeline_tes
import argparse
import traceback
from pathlib import Path

import numpy as np
import rasterio
from loguru import logger
from tqdm import tqdm

from data_pipeline.utils.path_utils import fetch_data_pair


def validate_tiffs(cube_path: Path, annotation_path: Path) -> tuple[bool, bool]:
    """
    Check two GeoTIFFs for identical shape and the absence of NaN pixels.

    Parameters
    ----------
    cube_path, annotation_path :
        Paths to the TIFF files that need to be compared.

    Returns
    -------
    same_shape :
        True  -> `cube.shape == annotation.shape`
        False -> shapes differ.
    same_nan_count :
        True  -> both rasters contain **zero** NaN pixels.
        False -> at least one raster contains NaNs (or NaN counts differ).

    Notes
    -----
    - For integer rasters that use a *nodata* value instead of NaNs, replace the
      `np.isnan(...)` tests with a comparison against `dataset.nodata` or use
      `dataset.read(1, masked=True).mask.any()`.
    """
    cube_path = Path(cube_path)
    annotation_path = Path(annotation_path)

    with (
        rasterio.open(cube_path) as cube,
        rasterio.open(annotation_path) as ann,
    ):
        same_shape = cube.shape == ann.shape
        cube_nan_count = np.isnan(cube.read(1)).sum()
        # Annotation is supposed to be uint8 because of well-defined classes
        # Therefore we need to use the nan mask to verify that the annotation is equal to zero there
        cube_arr = cube.read(1)  # float → may contain NaNs
        ann_arr = ann.read(1)  # uint8  → class ids (0, 1, 2, ...)

        cube_nan_mask = np.isnan(cube_arr)
        cube_nan_count = cube_nan_mask.sum()

        # Everywhere the cube is NaN the annotation must be 0
        ann_bad_count = (ann_arr[cube_nan_mask] != 0).sum()
        logger.debug(
            f'NaN count in cube: {cube_nan_count}  |  annotation non-zero @ cube-NaN: {ann_bad_count}',
        )

        same_nan_count = ann_bad_count == 0
    return same_shape, same_nan_count


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-imgpath', type=Path, required=True)
        args = parser.parse_args()
        if not args.imgpath.exists():
            raise FileNotFoundError('Img path not found')

        prod_names = [p.name for p in list(Path(args.imgpath).glob('*'))]
        prods_count = len(prod_names)
        logger.info(f'Found {prods_count}')

        same_shape_count = 0
        same_nan_count = 0
        for prod_name in tqdm(prod_names):
            if prod_name == 'hyperfield1a_L1B_20250319T031020':
                cube_path, annotation_path = fetch_data_pair(
                    root_dir=args.imgpath,
                    prod_name=prod_name,
                )

                same_shape_flag, same_nan_flag = validate_tiffs(
                    cube_path, annotation_path
                )
                same_shape_count += same_shape_flag
                same_nan_count += same_nan_flag

                logger.info(
                    f'Found with same shape {same_shape_count}/{prods_count}'
                )
                logger.info(
                    f'found with same NaN values {same_nan_count}/{prods_count}'
                )

    except Exception:
        logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
