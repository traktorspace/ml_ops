import getpass
import io
import platform
import sys
import traceback
import warnings
from datetime import (
    datetime,
)
from pathlib import (
    Path,
)
from time import (
    perf_counter,
)

import hydra
from dotenv import (
    dotenv_values,
)
from loguru import (
    logger,
)
from omegaconf import (
    DictConfig,
    OmegaConf,
)
from psycopg import Connection
from rasterio.errors import (
    NotGeoreferencedWarning,
)
from tqdm import tqdm

from annotation_pipeline.utils.db_utils import exec_query, init_connection
from annotation_pipeline.utils.encord_utils import (
    fetch_annotations_duplicates,
    fetch_annotations_from_workflow_stages,
    fetch_client,
    fetch_project,
)
from annotation_pipeline.utils.gcloud_utils import fetch_google_bucket
from annotation_pipeline.utils.hydra_helpers import (
    load_attr,
)
from annotation_pipeline.utils.job_processor import JobPushProcessor
from annotation_pipeline.utils.slack_utils import (
    post_new_message_and_get_thread_id,
    upload_file_to_channel,
    wrap_msg_with_project_name,
)

warnings.filterwarnings(
    'ignore',
    category=NotGeoreferencedWarning,
)


@hydra.main(
    version_base='1.3',
    config_path='../configs/cloud_annotation',
    config_name='data_push',
)
def main(
    cfg: DictConfig,
):
    projname: str = 'Unknown'
    env: dict = {}
    db_conn: Connection | None = None
    try:
        OmegaConf.register_new_resolver(
            'as_path',
            lambda x: Path(x),
        )

        # Check for missing keys
        missing_keys: set[str] = OmegaConf.missing_keys(cfg)
        if missing_keys:
            raise RuntimeError(f'Got missing keys in config:\n{missing_keys}')

        logger.info(
            'Script configuration:\n',
            OmegaConf.to_yaml(cfg),
        )
        noisy_pkgs: tuple[str, ...] = tuple(cfg.logging.noisy_pkgs)

        def is_noisy(
            record,
        ):
            return any(record['name'].startswith(pkg) for pkg in noisy_pkgs)

        env = dotenv_values(cfg.secrets.dotenv)
        cfg.logging.logdir.mkdir(
            parents=True,
            exist_ok=True,
        )
        logger.remove()
        logger.add(
            sys.stderr,
            format=cfg.logging.console_fmt,
            colorize=True,
            filter=lambda record: not is_noisy(record),
        )
        logger.add(
            cfg.logging.logdir / '{time:YYYYMMDD_HHmmss}.log',
            format=cfg.logging.file_fmt,
            level='INFO',
            filter=lambda record: not is_noisy(record),
        )
        projname = cfg['project_name']
        db_conn = init_connection(env)

        encord_client = fetch_client(Path(str(env['ENCORD_PRIVATE_KEY_PATH'])))
        encord_project = fetch_project(
            client=encord_client,
            encord_project_hash=str(env['ENCORD_CLOUD_PROJECT_HASH']),
        )
        all_encord_annotations = fetch_annotations_from_workflow_stages(
            project=encord_project
        )

        # TODO: ADD ANOTHER CHECK OVER THE CLOUD ANNOTATION DB
        if fetch_annotations_duplicates(all_encord_annotations):
            raise ValueError('Found duplicates in the annotations!')

        all_encord_annotations_unpacked = set().union(
            *all_encord_annotations.values()
        )

        # When pulling from db we expect to have an id and a path for each product
        query_result = set(
            exec_query(
                db_conn,
                load_attr(cfg.criteria.prod_selection_query),
                params=cfg.criteria.query_params,
            )
            or []
        )

        existing_products = [r for r in query_result if Path(r[1]).exists()]
        existing_products_paths = set(
            Path(r[1]).name for r in existing_products
        )
        logger.info(
            f'Found {len(all_encord_annotations_unpacked)} products on encord'
        )
        logger.info(
            f'Found  {len(existing_products_paths)} existing products fetched from the archive'
        )
        machine_id = f'{getpass.getuser()}@{platform.node()}'
        uploadable_products = (
            existing_products_paths - all_encord_annotations_unpacked
        )
        n_uploadable_products = len(uploadable_products)
        perc_uploadable_products = round(
            n_uploadable_products / len(existing_products_paths) * 100, 2
        )
        logger.info(
            f'{n_uploadable_products} products can be uploaded ({perc_uploadable_products}%).'
        )
        post_new_message_and_get_thread_id(
            text=wrap_msg_with_project_name(
                msg=f'Found *{n_uploadable_products}/{len(existing_products_paths)} ({perc_uploadable_products}%)*  '
                f'on *{machine_id}* that could be uploaded on Encord.\n'
                f'Using the following parameters:```{"\n".join(f"{k}: {v}" for k, v in cfg.criteria.query_params.items())}```',
                projname=projname,
            ),
            slack_bot_token=env['SLACK_OAUTH'],
            channel_id=env['SLACK_CHANNEL'],
        )

        dry_run: bool = cfg.job_processor.dry_run
        processor = JobPushProcessor(
            dst_bucket=fetch_google_bucket(
                env['ANNOTATIONS_GOOGLE_CLOUD_PROJECT'],
                env['ANNOTATIONS_GOOGLE_CLOUD_BUCKET_NAME'],
            ),
            bucket_root_folder=cfg.job_processor.bucket_root_folder,
            dst_bucket_folder=cfg.job_processor.dst_bucket_folder,
            db_conn=db_conn,
            q_success=load_attr(cfg.job_processor.query_success),
            q_failure=load_attr(cfg.job_processor.query_failure),
            allow_overwriting=cfg.job_processor.allow_overwriting,
            dry_run=dry_run,
        )

        start = perf_counter()

        # Run logic
        loop_products = [
            job
            for job in existing_products
            if Path(job[1]).name in uploadable_products
        ]
        for idx, (prod_id, prod_path) in enumerate(tqdm(loop_products)):
            try:
                if dry_run:
                    logger.info(
                        f'[DRY-RUN] would process job id={prod_id} : {prod_path}'
                    )
                    continue
                else:
                    processor(
                        prod_id,
                        prod_path,
                    )
            except Exception:
                if not dry_run:  # suppress UPDATE in dry run
                    exec_query(
                        db_conn,
                        processor.q_failure,
                        params={
                            'exception_message': str(traceback.format_exc()),
                            'annotation_id': prod_id,
                        },
                    )

        processor.finalize()
        elapsed = perf_counter() - start
        time_msg = f'Total time: {elapsed:.1f} s\n({len(existing_products) / elapsed:.2f} jobs/s)'
        logger.info(processor.stats.summary())
        stamp = datetime.now().strftime('%Y/%m/%d-%H:%M:%S')

        if not dry_run:
            success_msg = (
                f'✅ *Processing completed!*\n\n'
                f'⏰ *{stamp}*\n\n'
                f'(RGB, NIR) pairs have been stored in the remote bucket {cfg.job_processor.dst_bucket_folder}\n'
                f'Analyzed {len(loop_products)} products\n{time_msg}'
            )
            thread_id = post_new_message_and_get_thread_id(
                text=wrap_msg_with_project_name(
                    msg=success_msg,
                    projname=projname,
                ),
                slack_bot_token=str(env['SLACK_OAUTH']),
                channel_id=str(env['SLACK_CHANNEL']),
            )
            buffers = processor.stats.as_text_buffers()
            for name, filebuf in buffers.items():
                upload_file_to_channel(
                    buffer=filebuf,
                    filename=filebuf.name,
                    token=str(env['SLACK_OAUTH']),
                    channel_id=str(env['SLACK_CHANNEL']),
                    initial_comment='',
                    thread_ts=thread_id,
                )
            with open(processor.encord_payload_dst, 'rb') as fp:
                buf = io.BytesIO(fp.read())
                buf.seek(0)
                buf.name = 'payload.json'

                upload_file_to_channel(
                    buffer=buf,
                    filename=buf.name,
                    token=str(env['SLACK_OAUTH']),
                    channel_id=str(env['SLACK_CHANNEL']),
                    initial_comment='JSON payload that you need to upload on encord',
                    thread_ts=thread_id,
                )

    except Exception as e:
        trace_dump = traceback.format_exc()
        stamp = datetime.now().strftime('%Y/%m/%d-%H:%M:%S')
        logger.error(trace_dump)
        post_new_message_and_get_thread_id(
            text=wrap_msg_with_project_name(
                msg=f'❌ Error occured!\n⏰ *{stamp}*\n```\n{e}```',
                projname=projname,
            ),
            slack_bot_token=str(env['SLACK_OAUTH']),
            channel_id=str(env['SLACK_CHANNEL']),
        )
    finally:
        if db_conn is not None:
            try:
                db_conn.close()
            except Exception:
                logger.warning('Failed to close db connection!')


if __name__ == '__main__':
    main()
