# Send pair of image + annotation to slack chat

# python3 scripts/send_plot_to_chat.py
#  -dotenv=/home/tommaso.canova/repos/encord_scripts/data_pipeline/configs/cloud_annotation/.env
#  -imgpath=/bigdata/datasets_analytics/clouds/hf1a_reflectance/pipeline_test -prodname=hyperfield1a_L1B_20250608T073649

# If using a file
# python3 scripts/send_plot_to_chat.py
# -dotenv=/home/tommaso.canova/repos/encord_scripts/data_pipeline/configs/cloud_annotation/.env
# -imgpath=/bigdata/datasets_analytics/clouds/hf1a_reflectance/pipeline_test -prodlist=new_processed.txt
import matplotlib

matplotlib.use('Agg')
import argparse
import random
import traceback
from pathlib import Path

from dotenv import dotenv_values
from loguru import logger

from data_pipeline.utils.path_utils import fetch_data_pair
from data_pipeline.utils.slack_utils import (
    post_new_message_and_get_thread_id,
    upload_file_to_channel,
)
from data_pipeline.utils.viz_utils import build_cube_and_mask_preview


def fetch_and_set_plots_to_chat(
    prod_name: str,
    img_path: str,
    slack_token: str,
    slack_channel_id: str,
    overlay_alpha: float = 0.5,
    dpi: int = 300,
    thread_ts: str | None = None,
) -> float:
    """
    Build a cube / annotation preview for a product and push it to Slack.

    Parameters
    ----------
    prod_name :
        Name of the product to preview.
    img_path :
        Directory containing the cube + annotation files.
    slack_token :
        Slack OAuth token.
    slack_channel_id :
        Slack channel (ID) where the image will be posted.
    overlay_alpha :
        Alpha value for overlay used in `build_cube_and_mask_preview`.
    dpi :
        DPI setting used when rendering the preview image.
    thread_ts :
        If provided, the file is posted as a reply to this
        parent-message timestamp (i.e. inside that thread).
        Requires `channel_id` to be set as well.

    Returns
    -------
    Returns the thread timestamp to allow thread posting
    """
    # 1. Locate cube + annotation
    cube_path, annotation_path = fetch_data_pair(
        root_dir=img_path,
        prod_name=prod_name,
    )

    # 2. Build preview buffer
    buf = build_cube_and_mask_preview(
        cube_path=cube_path,
        annotation_path=annotation_path,
        overlay_alpha=overlay_alpha,
        dpi=dpi,
    )

    # 3. Push to Slack
    logger.info(f'Sending images for product {prod_name}')

    upload_file_to_channel(
        buffer=buf,
        filename='plot.png',
        token=slack_token,
        channel_id=slack_channel_id,
        title=prod_name,
        initial_comment=(
            f'Cube, Mask, Overlay for product `{prod_name}`\n'
            f'Location: *{img_path}*'
        ),
        thread_ts=thread_ts,
    )

    return thread_ts


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-dotenv', type=Path, required=True)
        parser.add_argument('-imgpath', type=Path, required=True)
        parser.add_argument('-prodname', type=str, required=False)
        parser.add_argument('-prodlist', type=Path, required=False)
        parser.add_argument(
            '-prodslist',
            type=Path,
            required=False,
            help='Provide a txt file with a list of products you want to see plot for',
        )
        args = parser.parse_args()

        if not args.dotenv.exists():
            raise FileNotFoundError('Dotenv file not found')

        if not args.imgpath.exists():
            raise FileNotFoundError('Img path not found')

        env = dotenv_values(args.dotenv)

        if args.prodname is None and args.prodlist is None:
            if Path(args.imgpath).exists():
                logger.info(
                    'No product name specified, picking up a random image from the images directory'
                )
                args.prodname = random.choice(
                    [p.name for p in list(Path(args.imgpath).glob('*'))]
                )
            else:
                raise FileNotFoundError(
                    f"{args.imgpath} imagepath doesn't exist"
                )

        thread_ts = None
        if (
            args.prodlist is not None
            and Path(args.prodlist).exists()
            and Path(args.imgpath).exists()
        ):
            with open(args.prodlist) as file:
                prods = file.read().splitlines()

        elif args.prodname is not None:
            prods = [args.prodname]

        else:
            raise ValueError('Wrong combination of parameters!')

        # If thread_ts is not defined we will post a message before sending the cubes
        thread_ts = post_new_message_and_get_thread_id(
            text=f'Requested [Cube, Mask, Overlay] for {len(prods)} product'
            f'{"s" if len(prods) > 1 else ""}.\n'
            'Images can be found in the thread 👇',
            slack_bot_token=env['SLACK_OAUTH'],
            channel_id=env['SLACK_CHANNEL'],
        )
        for idx, prod_name in enumerate(prods):
            fetch_and_set_plots_to_chat(
                prod_name=prod_name,
                img_path=args.imgpath,
                slack_token=env['SLACK_OAUTH'],
                slack_channel_id=env['SLACK_CHANNEL'],
                thread_ts=thread_ts,
            )
        logger.success('Upload completed!')

    except Exception:
        logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
