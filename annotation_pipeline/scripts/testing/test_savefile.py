import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio


def save_first_band_as_png(
    src_path: str | Path, dst_path: str | Path | None = None
) -> Path:
    """
    Read *src_path* (any GDAL-supported raster), take band-1, and write it
    as a grayscale PNG.

    Parameters
    ----------
    src_path : str | Path
        Path to the input raster.
    dst_path : str | Path | None, optional
        Where to save the PNG. Defaults to the same name with ``.png`` suffix.

    Returns
    -------
    Path
        The absolute path of the written PNG.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path) if dst_path else src_path.with_suffix('.png')

    # ------------------------------------------------------------------ read
    with rasterio.open(src_path) as src:
        band = src.read(1)

    # ----------------------------------------------------------- normalise
    band = band.astype(np.float32)
    vmin, vmax = np.nanmin(band), np.nanmax(band)
    if vmax > vmin:
        band = (band - vmin) / (vmax - vmin)  # range 0–1
    band = (band * 255).astype(np.uint8)

    # ------------------------------------------------------------------ plot
    plt.figure(figsize=(6, 6), dpi=150)
    plt.imshow(band, cmap='gray')
    plt.axis('off')
    plt.savefig(dst_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f'PNG written to {dst_path.resolve()}')
    return dst_path.resolve()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: python test_savefile.py <raster.tif> [output.png]')

    tif_path = sys.argv[1]
    out_png = sys.argv[2] if len(sys.argv) > 2 else None
    save_first_band_as_png(tif_path, out_png)
