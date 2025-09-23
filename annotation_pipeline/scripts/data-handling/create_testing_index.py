# python create_testing_index.py /bigdata/cloud_test/hf1a_cloud_dataset \
# --write-csv --write-plots --out-dir results \
# --n-samples 50 --max-cloudy 5 --max-clear 8

import argparse
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
    Randomly draw *n_samples* rows from *df* with the following limits:
      • at most *max_cloudy* rows whose total_cloud_coverage ≥ 90
      • at most *max_clear* rows whose Fill/Clear ≥ 99
    Remaining rows are filled at random from the rest.
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
) -> None:
    """
    Depending on the flags, create a CSV and/or one or two figures.
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
