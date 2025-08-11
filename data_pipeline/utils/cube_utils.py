from pathlib import Path

import cv2
import numpy as np
import rasterio
from frame_aligner import find_homographies, initialize_aligners
from frame_aligner.methods import BaseAligner
from kuva_metadata.sections_l0 import AlignmentAlgorithm
from PIL import Image


def _stretch_single_band_contrast_array(
    band_array, lower_quantile=0.01, upper_quantile=0.995
):
    """
    Contrast-stretch a single band using NumPy only.
    - NaN values are converted to 0 and treated as nodata.
    - Values ≤ 0 stay unchanged after stretching.
    """
    band_array = np.nan_to_num(band_array, nan=0.0)
    mask = band_array > 0
    masked = band_array[mask]

    if masked.size:
        lower = np.quantile(masked, lower_quantile)
        upper = np.quantile(masked, upper_quantile)
        span = upper - lower

        if span == 0:
            stretched = np.zeros_like(band_array, dtype=np.uint8)
            stretched[mask] = band_array[mask].astype(np.uint8)
            return stretched

        with np.errstate(divide='ignore', invalid='ignore'):
            stretched = (band_array - lower) / span * 255.0

        stretched = np.clip(stretched, 0, 255)
        stretched[~mask] = 0
        return stretched.astype(np.uint8)

    return band_array.astype(np.uint8)


def extract_bands(
    filepath: Path,
    save_product: bool = False,
    dst_path: Path | None = None,
    stretch_contrast: bool = False,
    red_band_index: int = 10,
    green_band_index: int = 5,
    blue_band_index: int = 1,
    nir_band_index: int = 22,
    verbose: bool = False,
):
    """
    Extract RGB + NIR bands from a multiband raster.

    Parameters
    ----------
    filepath : str | Path
        Source raster file.
    save_product : bool
        If True, write two PNGs (`*_rgb.png`, `*_nir.png`) to `dst_path`.
    dst_path : Path | None
        Base folder for outputs (required when `save_product=True`).
    stretch_contrast : bool
        Apply 1-2 % contrast stretching to each band.
    red_band_index / green_band_index / blue_band_index / nir_band_index : int
        1-based band positions inside the raster.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (rgb_array, nir_array)
    """

    def ndarray2png(ndarray: np.ndarray):
        ndarray = np.nan_to_num(ndarray, nan=0.0, posinf=1.0, neginf=0.0)
        return (255 * np.clip(ndarray, 0.0, 1.0)).astype(np.uint8)

    if isinstance(filepath, str):
        filepath = Path(filepath)

    with rasterio.open(filepath) as raster:
        band_indices = {
            'R': red_band_index,
            'G': green_band_index,
            'B': blue_band_index,
            'NIR': nir_band_index,
        }

        bands = {}
        for key, idx in band_indices.items():
            arr = raster.read(idx)
            if stretch_contrast:
                arr = _stretch_single_band_contrast_array(arr)
            bands[key] = arr

        rgb_array = np.dstack([bands['R'], bands['G'], bands['B']])
        nir_array = bands['NIR']

    if save_product:
        if dst_path is None:
            raise ValueError(
                '`dst_path` must be provided when `save_product=True`.'
            )

        parent_folder = filepath.parent.name
        out_dir = dst_path / parent_folder
        out_dir.mkdir(parents=True, exist_ok=True)

        rgb_path = out_dir / f'{parent_folder}_rgb.png'
        nir_path = out_dir / f'{parent_folder}_nir.png'

        Image.fromarray(ndarray2png(rgb_array)).save(rgb_path)
        Image.fromarray(ndarray2png(nir_array), mode='L').save(nir_path)

        if verbose:
            print(f'RGB saved to {rgb_path}')
            print(f'NIR saved to {nir_path}')


def get_cloud_labels_dict() -> dict[str, int]:
    """Returns a dictionary containing the integer value
    for each class of the cloud annotation"""

    return {
        'Fill': 0,
        'Cloud_Shadow': 64,
        'Clear': 128,
        'Thin_Cloud': 192,
        'Cloud': 255,
    }


def to_uint16(
    arr: np.ndarray,
    in_min: float | None = None,
    in_max: float | None = None,
    nan_value: int = 0,
) -> np.ndarray:
    """
    Linearly rescale any numeric array to uint16.

    Parameters
    ----------
    arr:
        Input array (any dtype). NaNs are handled.
    in_min/in_max:
        Input range. Inferred from finite values if omitted.
    nan_value:
        Value assigned where `arr` is NaN.

    Returns
    -------
    np.ndarray (uint16) of same shape as `arr`.
    """
    UINT16_MAX = np.iinfo(np.uint16).max
    arr = arr.astype(np.float32, copy=False)

    # Derive range from finite values if not provided
    if in_min is None or in_max is None:
        finite = arr[np.isfinite(arr)]
        in_min = float(in_min if in_min is not None else finite.min())
        in_max = float(in_max if in_max is not None else finite.max())

    if in_max <= in_min:
        raise ValueError('in_max must be greater than in_min')

    scale = UINT16_MAX / (in_max - in_min)
    out = np.clip((arr - in_min) * scale, 0, UINT16_MAX)
    out = np.nan_to_num(out, nan=nan_value)

    return out.astype(np.uint16)


def float32_to_uint16(
    arr_float32: np.ndarray,
    in_min: float | None = None,
    in_max: float | None = None,
    nan_value: int = 0,
) -> np.ndarray:
    """
    Rescale a float32 array to uint16.

    Parameters
    ----------
    arr_float32 : np.ndarray
        Input array with dtype float32.
    in_min / in_max : float, optional
        Input range. Inferred when omitted.
    nan_value : int, default 0
        Value written wherever `arr_float32` is NaN.
    """
    return to_uint16(
        arr_float32, in_min=in_min, in_max=in_max, nan_value=nan_value
    )


def rescale_u8_to_u16(
    a8: np.ndarray,
    in_min: int = 0,
    in_max: int = 255,
    nan_value: int = 0,
) -> np.ndarray:
    """
    Fast uint8 → uint16 rescaling.

    Parameters
    ----------
    a8 : np.ndarray
        Input array with dtype uint8.
    in_min : int, default 0
    in_max : int, default 255
    nan_value : int, default 0
        Value written wherever `a8` is NaN (rare for uint8).

    Returns
    -------
    np.ndarray
        Rescaled array of dtype uint16.
    """
    return to_uint16(a8, in_min=in_min, in_max=in_max, nan_value=nan_value)


def get_aligner(
    aligner_name: str = 'roma', aligner_parameters: dict = {'device': 'cuda:4'}
):
    # Initialize aligners to calculate homography transformations
    prealignment_mode = [
        AlignmentAlgorithm(name=aligner_name, parameters=aligner_parameters),
    ]
    aligners = initialize_aligners(prealignment_mode)
    return aligners


def apply_nan_mask_to_annotation(
    annotation: np.ndarray, src_raster: np.ndarray, fill_value: int = 0
) -> np.ndarray:
    """
    Replace values in `annotation` by `fill_value` wherever `src_raster`
    contains NaNs.

    Supported shapes
    ----------------
    1. 3-D / 3-D
       annotation : (C, W, H) - normally C == 1
       src_raster : (N, W, H)

    2. 2-D / 2-D
       annotation : (W, H)
       src_raster : (W, H)

    Any other dimensionality mix raises ``ValueError``.

    Parameters
    ----------
    annotation
        Target array whose values will be overwritten.
    src_raster
        Source raster whose NaNs define the mask.
    fill_value
        Value that replaces `annotation` where the mask is True.

    Returns
    -------
    ndarray
        Same shape as `annotation`, with NaN-masked values replaced.
    """

    def _check_spatial_dims(
        a_shape: tuple[int, ...], b_shape: tuple[int, ...]
    ) -> None:
        """Raise if (W, H) dimensions differ."""
        if a_shape[-2:] != b_shape[-2:]:
            raise ValueError('annotation and src_raster must share (W, H)')

    # 0. Basic sanity checks
    if annotation.ndim not in (2, 3) or src_raster.ndim not in (2, 3):
        raise ValueError(
            'Only 2-D or 3-D arrays are supported '
            f'(got annotation.ndim={annotation.ndim}, src_raster.ndim={src_raster.ndim})'
        )

    # 1. Verify spatial sizes (W, H) match
    _check_spatial_dims(annotation.shape, src_raster.shape)

    # 2. Build a 2-D mask (W, H)
    if src_raster.ndim == 3:  # e.g. (C, W, H)
        nan_mask = np.isnan(src_raster).any(axis=0)
    else:  # (W, H)
        nan_mask = np.isnan(src_raster)

    # 3. Broadcast the mask to `annotation`'s shape and apply
    if annotation.ndim == 3:  # (C, W, H)
        nan_mask = nan_mask[None, ...]  # -> (1, W, H) or broadcast to C
    return np.where(nan_mask, fill_value, annotation)


def align_label_with_new_reference(
    old_frame_path: Path,
    new_frame_path: Path,
    aligners: list[BaseAligner],
    old_label: np.ndarray,
    old_frame_band_n: int = 1,
    new_frame_band_n: int = 10,
):
    # Open old frame as
    with rasterio.open(old_frame_path) as old_src:
        old_frame = old_src.read(old_frame_band_n)
        old_frame = rescale_u8_to_u16(old_frame)

    # Open current tif file (latest pipeline generated version)
    with rasterio.open(new_frame_path) as new_src:
        new_frame = new_src.read(new_frame_band_n)
        new_frame = float32_to_uint16(new_frame)
        new_frame_profile = new_src.profile

    cube = [old_frame, new_frame]
    masks = [None for frame in cube]
    frames_metadata = [None for _ in cube]
    frames_camera = [None for _ in cube]

    # Calculate homography transformations
    _, homographies = find_homographies(
        cube=cube,
        masks=masks,
        frames_metadata=frames_metadata,
        frames_camera=frames_camera,
        ref_frame_index=0,
        aligners=aligners,
    )

    # Get the homography to move from old to new reference
    H_old2new = np.linalg.inv(homographies[1])

    # Fetch label
    old_label = rescale_u8_to_u16(old_label)
    # Warp label based on homography
    new_label = cv2.warpPerspective(
        src=old_label,
        M=H_old2new,
        dsize=new_frame.shape[:2][::-1],
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
    ).astype(np.uint8)

    new_frame_profile['count'] = 1

    return new_label, new_frame_profile


def store_annotation(
    annotation_tensor: np.ndarray, annotation_profile: dict, dst_path: Path
):
    with rasterio.open(dst_path, 'w', **annotation_profile) as dst:
        dst.write(annotation_tensor.astype(annotation_profile['dtype']), 1)


def mismatched_products(
    root: Path, frame: str = 'L1B.tif', ann: str = 'L1B_annotation.tif'
) -> list[str]:
    """
    Find products whose annotation and reference frame have different raster sizes.

    Parameters
    ----------
    root:
        Directory that contains one sub-directory per product.
        Each product folder is expected to hold the two TIFF files to compare.
    frame:
        File-name of the reference raster inside every product folder.
    ann:
        File-name of the annotation raster inside every product folder.

    Returns
    -------
    list[str]
        Names of the product folders where ``frame`` and ``ann`` differ
        in either height or width. Products missing one of the two files
        are silently skipped.

    Notes
    -----
    The function opens each pair of TIFFs with *rasterio* and compares their
    ``height`` and ``width`` attributes, which avoids reading the full pixel
    array into memory.

    Examples
    --------
    >>> from pathlib import Path
    >>> bad = mismatched_products(Path("/data/test_dst"))
    >>> len(bad)
    3
    >>> bad
    ['prod_0021', 'prod_0045', 'prod_0107']
    """
    bad: list[str] = []
    for prod_dir in root.iterdir():
        f, a = prod_dir / frame, prod_dir / ann
        if not (f.exists() and a.exists()):
            continue
        with rasterio.open(f) as ff, rasterio.open(a) as aa:
            if (ff.height, ff.width) != (aa.height, aa.width):
                bad.append(prod_dir.name)
    return bad
