import traceback
from pathlib import Path
from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from loguru import logger
from tqdm import tqdm

VALUE2CLASS = {
    0: 'Fill/Clear',
    128: 'Fill/Clear',
    64: 'Cloud Shadow',
    192: 'Thin Cloud',
    255: 'Cloud',
}


def empty_counter() -> dict[str, int]:
    """Return an empty counter for every class."""
    return {cls: 0 for cls in set(VALUE2CLASS.values())}


def value_to_class(value: int) -> str:
    try:
        return VALUE2CLASS[value]
    except KeyError as e:
        raise ValueError(f'Unexpected label value {value}') from e


def save_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    """
    1. Write a CSV with the statistics.
    2. Produce one figure (4-panel histogram) for the per-class percentages.
    3. Produce one figure (single histogram) for the overall cloud-coverage
       percentages.

    Histograms use *relative* frequencies, i.e. the y-axis shows the share of
    products (%) whose value falls into a given bin.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ CSV ---
    csv_path = out_dir / 'cloud_statistics.csv'
    df.to_csv(csv_path)
    logger.info(f'CSV written to {csv_path}')

    # common objects -----------------------------------------------------------
    n_products = len(df)
    weights = np.full(
        n_products, 100 / n_products
    )  # every product counts equally
    cmap = plt.get_cmap('tab10')  # colour palette

    # ---------------------------------------------------------------- FIG 1 ---
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

    # ---------------------------------------------------------------- FIG 2 ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        df['total_cloud_coverage'].values,
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


def main():
    try:
        root = Path('/bigdata/cloud_test/hf1a_cloud_dataset')
        annotations = list(root.rglob('L1B_annotation.tif'))

        rows: list[dict[str, float]] = []

        statistics: dict[str, dict[str, dict[str, float | int]]] = {}

        for ann in tqdm(annotations, desc='Processing masks'):
            with rasterio.open(ann) as src:
                mask = src.read(1)

            # pixel counts per raw value
            values, counts = np.unique(mask, return_counts=True)

            # convert to class counts (0 & 128 collapse into “Fill/Clear”)
            class_counts = empty_counter()
            for v, c in zip(values, counts):
                cls = value_to_class(int(v))
                class_counts[cls] += int(c)

            # convert counts to percentages
            total_px = sum(class_counts.values())
            class_pct = {
                cls: round(cnt / total_px * 100, 3)
                for cls, cnt in class_counts.items()
            }
            overall_cloud_coverage = round(
                sum(v for k, v in class_pct.items() if k != 'Fill/Clear'),
                3,
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
                }
            )

        pprint(statistics)
        df = pd.DataFrame(rows).set_index('product')
        save_outputs(df, out_dir=Path('output'))
        logger.info('Operation completed.')

    except Exception:
        logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
