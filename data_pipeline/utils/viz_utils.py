from io import BytesIO
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def map_classes_to_values(annotation: np.ndarray, dict_map: dict[int, int]):
    """Map a ndarray using a custom dict map"""
    for k, v in dict_map.items():
        annotation[annotation == k] = v
    return annotation


def build_cube_and_mask_preview(
    cube_path: Path,
    mask_path: Path,
    mask_plot_title: str = 'Annotation',
    overlay_alpha: float = 0.5,
    dpi: int = 300,
    legend_map: dict = {
        0: 'Fill/Clear',
        1: 'Cloud Shadow',
        2: 'Thin Cloud',
        3: 'Cloud',
    },
    mapping_values={
        128: 0,
        64: 1,
        192: 2,
        255: 3,
    },
) -> BytesIO:
    """
    Create a side-by-side visual preview of a data cube and its annotation mask.

    The figure contains three panels:

        1. The first band of the cube in grayscale.
        2. The annotation mask after remapping its raw pixel values to class
           indices.
        3. An overlay of the annotation (with configurable alpha) on the cube.

    A legend is automatically generated from `legend_map` and placed to the
    right of the panels.  The suptitle is set to the name of the parent
    directory containing `cube_path`.

    Parameters
    ----------
    cube_path:
        Path to the multi-band raster file.  Only band 1 is shown.
    mask_path:
        Path to the annotation or cloud mask raster.  Its raw values are converted to class
        indices by applying `mapping_values`.
    overlay_alpha:
        Transparency of the annotation when it is overlaid on the cube in the
        third panel.  Must be in the range 0-1.
    dpi:
        Resolution (dots per inch) of the generated figure.
    legend_map:
        Maps class indices (i.e. values **after** applying `mapping_values`)
        to the human-readable class names shown in the legend.  The colours are
        taken from the *viridis* colormap and are assigned in ascending order
        of the keys.
    mapping_values:
        Maps raw pixel values found in the annotation raster to the class
        indices used by `legend_map`.  Any value not present in this mapping is
        left unchanged.

    Returns
    -------
    io.BytesIO
        A buffer positioned at byte 0 that contains a PNG representation of the
        figure.  The caller is responsible for writing it to disk or further
        processing.

    Notes
    -----
    - The function closes the Matplotlib figure before returning, so it will
      not interfere with other plots in the current session.
    - The buffer can be saved to a file via
      `output_path.write_bytes(buf.getvalue())`
      or, equivalently, with `open(output_path, "wb").write(buf.getvalue())`.

    Examples
    --------
    >>> from pathlib import Path
    >>> buf = build_cube_and_mask_preview(
    ...     cube_path=Path("L1BXXX_cube.tif"),
    ...     annotation_path=Path("L1BXXX_annotation.tif"),
    ...     overlay_alpha=0.4,
    ... )
    >>> # Write the preview to disk
    >>> Path("preview.png").write_bytes(buf.getvalue())
    """

    cmap = plt.cm.viridis
    patches = [
        mpatches.Patch(color=cmap(i / max(legend_map.keys())), label=label)
        for i, label in legend_map.items()
    ]

    cube_path, annotation_path = Path(cube_path), Path(mask_path)

    with rasterio.open(cube_path) as src:
        cube = src.read(1)

    with rasterio.open(annotation_path) as src:
        annotation = map_classes_to_values(src.read(1), mapping_values)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=dpi)

    axes[0].imshow(cube, cmap='gray')
    axes[0].set_title('Cube (band 1)')
    axes[0].axis('off')

    axes[1].imshow(annotation, cmap='viridis')
    axes[1].set_title(mask_plot_title)
    axes[1].axis('off')

    axes[2].imshow(cube, cmap='gray')
    axes[2].imshow(annotation, cmap='viridis', alpha=overlay_alpha)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.legend(
        handles=patches,
        title='Legend',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        borderaxespad=0.0,
    )

    # Use common folder name as suptitle
    common_parent = cube_path.parent.name
    fig.suptitle(common_parent, fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Dump to buffer
    buf = BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return buf
