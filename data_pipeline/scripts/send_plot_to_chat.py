# python3 scripts/send_plot_to_chat.py
#  -dotenv=/home/tommaso.canova/repos/encord_scripts/data_pipeline/configs/cloud_annotation/.env
#  -imgpath=/bigdata/datasets_analytics/clouds/hf1a_reflectance/pipeline_test -prodname=hyperfield1a_L1B_20250319T030559
import matplotlib

matplotlib.use('Agg')
import argparse
import random
import traceback
from pathlib import Path

from dotenv import dotenv_values
from loguru import logger

from data_pipeline.utils.path_utils import fetch_data_pair
from data_pipeline.utils.slack_utils import upload_file_to_channel
from data_pipeline.utils.viz_utils import build_cube_annotation_preview


def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('-dotenv', type=Path, required=True)
        parser.add_argument('-imgpath', type=Path, required=True)
        parser.add_argument('-prodname', type=str, required=False)
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

        if args.prodname is None:
            if Path(args.imgpath).exists():
                args.prodname = random.choice(
                    [p.name for p in list(Path(args.imgpath).glob('*'))]
                )
            else:
                raise FileNotFoundError(
                    f"{args.imgpath} imagepath doesn't exist"
                )

        cube_path, annotation_path = fetch_data_pair(
            root_dir=args.imgpath,
            prod_name=args.prodname,
        )

        buf = build_cube_annotation_preview(
            cube_path=cube_path,
            annotation_path=annotation_path,
            overlay_alpha=0.5,
            dpi=300,
        )

        logger.info(f'Sending images for product {args.prodname}')

        upload_file_to_channel(
            buffer=buf,
            filename='plot.png',
            token=env['SLACK_OAUTH'],
            channel_id=env['SLACK_CHANNEL'],
            title=args.prodname,
            initial_comment=f'Cube, Mask, Overlay for product `{args.prodname}`\nLocation: *{args.imgpath}*',
        )

        logger.success('Upload completed!')

    except Exception:
        logger.exception(traceback.format_exc())


if __name__ == '__main__':
    main()
