# python create_testing_index.py /bigdata/cloud_test/hf1a_cloud_dataset \
# --write-csv --write-plots --out-dir results \
# --n-samples 50 --max-cloudy 5 --max-clear 8

import argparse
import traceback
from pathlib import Path
from pprint import pprint
from typing import cast

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from cartopy.mpl.geoaxes import GeoAxes
from loguru import logger
from shapely.geometry import Point
from tqdm import tqdm

VALUE2CLASS = {
    0: 'Fill/Clear',
    128: 'Fill/Clear',
    64: 'Cloud Shadow',
    192: 'Thin Cloud',
    255: 'Cloud',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Collect cloud-mask statistics for the HF1A dataset'
    )
    parser.add_argument(
        'root',
        type=Path,
        help='Root directory that contains the *L1B_annotation.tif* masks',
    )
    parser.add_argument(
        '--write-plots',
        action='store_true',
        help='Write the PNG histograms (default: off)',
    )
    parser.add_argument(
        '--write-csv',
        action='store_true',
        help='Write the CSV statistics file (default: off)',
    )
    parser.add_argument(
        '--out-dir',
        type=Path,
        default=Path('output'),
        help='Directory that receives the generated artefacts (default: ./output)',
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=None,
        help='Randomly keep only N products (default: keep all)',
    )
    parser.add_argument(
        '--max-cloudy',
        type=int,
        default=None,
        help='Max products with ≥ 90 % cloud coverage (default: unlimited)',
    )
    parser.add_argument(
        '--max-clear',
        type=int,
        default=None,
        help='Max products with ≥ 99 % Fill/Clear (default: unlimited)',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random-generator seed for sampling (default: 42)',
    )

    return parser.parse_args()


def empty_counter() -> dict[str, int]:
    """Return an empty counter for every class."""
    return {cls: 0 for cls in set(VALUE2CLASS.values())}


def value_to_class(value: int) -> str:
    try:
        return VALUE2CLASS[value]
    except KeyError as e:
        raise ValueError(f'Unexpected label value {value}') from e


def constrained_sample(
    df: pd.DataFrame,
    n_samples: int,
    max_cloudy: int | None,
    max_clear: int | None,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Randomly draw a specified number of rows from a DataFrame with constraints.

    Parameters
    ----------
    df :
        Input DataFrame containing cloud-mask statistics.
    n_samples :
        Total number of rows to sample.
    max_cloudy :
        Maximum number of rows with total_cloud_coverage ≥ 90.
    max_clear :
        Maximum number of rows with Fill/Clear ≥ 99.
    rng :
        Random number generator for reproducibility.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the sampled rows.
    """
    is_cloudy = df['total_cloud_coverage'] >= 90
    is_clear = df['Fill/Clear'] >= 99

    cloudy_df = df[is_cloudy]
    clear_df = df[is_clear]
    rest_df = df[~(is_cloudy | is_clear)]

    n_cloudy = min(
        len(cloudy_df), max_cloudy if max_cloudy is not None else len(cloudy_df)
    )
    n_clear = min(
        len(clear_df), max_clear if max_clear is not None else len(clear_df)
    )

    cloudy_sel = (
        cloudy_df.sample(n=n_cloudy, random_state=int(rng.integers(0, 2**32)))
        if n_cloudy
        else cloudy_df.iloc[:0]
    )
    clear_sel = (
        clear_df.sample(n=n_clear, random_state=int(rng.integers(0, 2**32)))
        if n_clear
        else clear_df.iloc[:0]
    )

    remaining = n_samples - len(cloudy_sel) - len(clear_sel)
    rest_sel = (
        rest_df.sample(
            n=min(remaining, len(rest_df)),
            random_state=int(rng.integers(0, 2**32)),
        )
        if remaining > 0
        else rest_df.iloc[:0]
    )

    sampled = pd.concat([cloudy_sel, clear_sel, rest_sel])

    # If still short, top-up with anything left
    if len(sampled) < n_samples:
        pool = df.drop(sampled.index)
        extra = (
            pool.sample(
                n=min(n_samples - len(sampled), len(pool)),
                random_state=int(rng.integers(0, 2**32)),
            )
            if not pool.empty
            else pool
        )
        sampled = pd.concat([sampled, extra])

    return sampled


def save_outputs(
    df: pd.DataFrame,
    out_dir: Path,
    write_csv: bool,
    write_plots: bool,
    annotations: list[Path],
) -> None:
    """
    Save outputs such as CSV files and plots based on the provided flags.

    Parameters
    ----------
    df :
        DataFrame containing cloud-mask statistics.
    out_dir :
        Directory where outputs will be saved.
    write_csv :
        Flag to indicate whether to save the statistics as a CSV file.
    write_plots :
        Flag to indicate whether to generate and save plots.
    annotations :
        List of paths to annotation files.

    Returns
    -------
    None
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Write csv
    if write_csv:
        csv_path = out_dir / 'cloud_statistics.csv'
        df.to_csv(csv_path)
        logger.info(f'CSV written to {csv_path}')

    # If plots are not required we can return early
    if not write_plots:
        return

    # Common objects for the two figures
    n_products = len(df)
    weights = np.full(n_products, 100 / n_products)
    cmap = plt.get_cmap('tab10')

    # Hist per class
    class_cols = ['Fill/Clear', 'Cloud Shadow', 'Thin Cloud', 'Cloud']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    axs = axs.flatten()

    for i, col in enumerate(class_cols):
        ax = axs[i]
        ax.hist(
            df[col].values,
            bins=20,
            weights=weights,
            color=cmap(i),
            edgecolor='black',
        )
        ax.set_title(col)
        ax.set_xlabel('Percentage of pixels')
        ax.set_ylabel('Products (%)')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)

    fig.suptitle(
        'Histogram of class percentages across products', fontsize=14, y=1.02
    )
    fig_path = out_dir / 'class_histograms.png'
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Class-histogram figure written to {fig_path}')

    # Overall cloud coverage hist
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        df['total_cloud_coverage'].values,  # type: ignore
        bins=20,
        weights=weights,
        color='steelblue',
        edgecolor='black',
    )
    ax.set_xlabel('Cloud coverage (%)')
    ax.set_ylabel('Products (%)')
    ax.set_title('Histogram of overall cloud coverage')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    plt.tight_layout()
    fig_path = out_dir / 'total_cloud_coverage_hist.png'
    fig.savefig(fig_path, dpi=300)
    plt.close(fig)
    logger.info(f'Cloud-coverage histogram written to {fig_path}')

    # Generate a heatmap of the world based on annotation coordinates
    coords = []
    for ann in annotations:
        with rasterio.open(ann) as src:
            # Extract the center of the raster as the coordinate in float degrees
            bounds = src.bounds
            lon = float((bounds.left + bounds.right) / 2)
            lat = float((bounds.top + bounds.bottom) / 2)
            coords.append((lon, lat))

    # Create a GeoDataFrame for the coordinates
    gdf = gpd.GeoDataFrame(
        {'geometry': [Point(lon, lat) for lon, lat in coords]},
        crs='EPSG:4326',  # WGS84
    )

    # Plot the heatmap
    fig = plt.figure(figsize=(14, 8))
    ax = cast(GeoAxes, plt.axes(projection=ccrs.PlateCarree()))
    ax.set_global()
    ax.add_feature(cfeature.LAND, edgecolor='black', facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.coastlines(resolution='110m', linewidth=0.8)
    gridlines = ax.gridlines(
        draw_labels=True, linewidth=0.5, color='gray', alpha=0.5, linestyle='--'
    )
    gridlines.top_labels = False
    gridlines.right_labels = False
    gridlines.xlabel_style = {'size': 10, 'color': 'gray'}
    gridlines.ylabel_style = {'size': 10, 'color': 'gray'}

    x, y = zip(
        *[
            (cast(Point, point).x, cast(Point, point).y)
            for point in gdf.geometry
        ]
    )
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=100)
    im = ax.imshow(
        heatmap.T,
        extent=[min(xedges), max(xedges), min(yedges), max(yedges)],
        origin='lower',
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        alpha=0.6,
    )

    cbar = fig.colorbar(im, ax=ax, orientation='vertical', shrink=0.7, pad=0.05)
    cbar.set_label('Frequency', fontsize=12)
    ax.set_title(
        'Heatmap of Annotation Locations', fontsize=14, fontweight='bold'
    )
    # Save the figure
    fig_path = out_dir / 'annotation_heatmap.png'
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    logger.info(f'Annotation heatmap written to {fig_path}')


def main():
    args = parse_args()

    try:
        root: Path = args.root.expanduser().resolve()
        annotations = list(root.rglob('L1B_annotation.tif'))

        rows: list[dict[str, float]] = []
        statistics: dict[str, dict[str, dict[str, float | int]]] = {}

        for ann in tqdm(annotations, desc='Processing masks'):
            with rasterio.open(ann) as src:
                mask = src.read(1)

            values, counts = np.unique(mask, return_counts=True)

            class_counts = empty_counter()
            for v, c in zip(values, counts):
                cls = value_to_class(int(v))
                class_counts[cls] += int(c)

            total_px = sum(class_counts.values())
            class_pct = {
                cls: round(cnt / total_px * 100, 3)
                for cls, cnt in class_counts.items()
            }
            overall_cloud_coverage = round(
                sum(v for k, v in class_pct.items() if k != 'Fill/Clear'), 3
            )

            product_name = str(ann.parent.stem)
            statistics[product_name] = {  # type: ignore
                'unique_vals': len(class_counts),
                'percentages': class_pct,
                'total_cloud_coverage': overall_cloud_coverage,
            }

            rows.append(
                {
                    'product': product_name,  # type: ignore
                    **class_pct,
                    'total_cloud_coverage': overall_cloud_coverage,
                    'unique_vals': len(class_counts),
                }
            )

        pprint(statistics)
        df = pd.DataFrame(rows).set_index('product')

        save_outputs(
            df,
            out_dir=args.out_dir,
            write_csv=args.write_csv,
            write_plots=args.write_plots,
            annotations=annotations,
        )

        if args.n_samples is not None and args.n_samples > 0:
            rng = np.random.default_rng(args.seed)
            sampled_df = constrained_sample(
                df,
                n_samples=args.n_samples,
                max_cloudy=args.max_cloudy,
                max_clear=args.max_clear,
                rng=rng,
            )

            args.out_dir.mkdir(parents=True, exist_ok=True)
            txt_path = args.out_dir / 'sampled_products.txt'
            txt_path.write_text('\n'.join(sampled_df.index) + '\n')
            logger.info(f'Sampled product list written to {txt_path}')

        logger.info('Operation completed.')

    except Exception:
        logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
