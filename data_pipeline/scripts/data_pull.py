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
from tqdm import (
    tqdm,
)

from data_pipeline.utils.cube_utils import (
    mismatched_products,
)
from data_pipeline.utils.db_utils import (
    exec_query,
    init_connection,
)
from data_pipeline.utils.encord_utils import (
    fetch_annotation_signed_url,
    fetch_annotations_duplicates,
    fetch_annotations_from_workflow_stages,
    fetch_client,
    fetch_project,
)
from data_pipeline.utils.gcloud_utils import BucketWrapper
from data_pipeline.utils.hydra_helpers import (
    load_attr,
)
from data_pipeline.utils.job_processor import (
    JobPullProcessor,
)
from data_pipeline.utils.path_utils import (
    parse_s3_path,
    tif_exists,
)
from data_pipeline.utils.slack_utils import (
    post_new_message_and_get_thread_id,
    upload_file_to_channel,
    wrap_msg_with_project_name,
)

warnings.filterwarnings(
    'ignore',
    category=NotGeoreferencedWarning,
)


def usage_msg(
    use_cloud_data_source: bool,
    save_only_annotation: bool,
    destination_path: str,
) -> str:
    """Return a human-friendly summary of what the pull just did."""
    origin = (
        '☁️ Data pulled from Google cloud!'
        if use_cloud_data_source
        else '💾 Data pulled from the local disk'
    )

    if save_only_annotation:
        detail = '🚨 Note: only annotations have been saved!\n'
    else:
        item = 'image' if use_cloud_data_source else 'image symlink'
        detail = (
            f'Pairs ({item}, annotation file) have been stored at '
            f'destination: {destination_path}\n'
        )

    return f'{origin}\n{detail}'


@hydra.main(
    version_base='1.3',
    config_path='../configs/cloud_annotation',
    config_name='data_pull',
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

        dry_run = cfg.job_processor.dry_run
        if dry_run:
            logger.warning(
                'DRY-RUN ENABLED - no permanent action will be taken!'
            )

        encord_client = fetch_client(Path(str(env['ENCORD_PRIVATE_KEY_PATH'])))
        encord_project = fetch_project(
            client=encord_client,
            encord_project_hash=str(env['ENCORD_CLOUD_PROJECT_HASH']),
        )
        all_encord_annotations = fetch_annotations_from_workflow_stages(
            project=encord_project
        )

        if fetch_annotations_duplicates(all_encord_annotations):
            raise ValueError('Found duplicates in the annotations!')

        USE_CLOUD_DATA_SOURCE: bool = cfg.job_processor.use_cloud_data_source
        projname = str(cfg['project_name'])
        db_conn = init_connection(env)
        completed_annotations = set(all_encord_annotations['Complete'])
        logger.info(
            f'Found {len(completed_annotations)} annotation marked as Completed on Encord'
        )

        products = (
            exec_query(
                db_conn,
                load_attr(cfg.criteria.prod_selection_query),
            )
            or []
        )

        logger.info(
            f'According to DB: Found {len(products)} products not downloaded yet'
        )
        completed_set = set(completed_annotations)
        # Keep only the rows whose product-id is in our completed list
        products_to_download = [
            row for row in products if row[1] in completed_set
        ]
        if USE_CLOUD_DATA_SOURCE:
            # TODO: implement a check before
            logger.warning(
                'Running with USE_CLOUD_DATA_SOURCE, the files will be fetched at runtime'
            )
        else:
            ready = [row for row in products_to_download if tif_exists(row)]
            missing_n = len(products_to_download) - len(ready)
            if missing_n:
                raise ValueError(
                    f'Only {len(ready)}/{len(products_to_download)} products are ready '
                    f'to be downloaded ({missing_n} original file(s) missing)'
                )

            if len(ready) == 0:
                warn_msg = '🤷‍♂️ No products are ready to be downloaded. Ending process gracefully 🪷'
                logger.warning(warn_msg)
                post_new_message_and_get_thread_id(
                    text=wrap_msg_with_project_name(
                        msg=f'```\n{warn_msg}```',
                        projname=projname,
                    ),
                    slack_bot_token=str(env['SLACK_OAUTH']),
                    channel_id=str(env['SLACK_CHANNEL']),
                )
            else:
                logger.info(f'{len(ready)} products are ready to be downloaded')
                products_to_download = ready

        # Limit products to download to specific amount for debugging purposes
        if cfg.debug_settings.limit_product_processed:
            products_to_download = products_to_download[
                : cfg.debug_settings.max_product_processed
            ]

        # Create a dictionary containing
        # db_id, prodname, s3_path, remote_annotation_path
        job_dict = {
            str(db_id): {
                'prodname': prodname,
                's3_path': parse_s3_path(
                    prodname=prodname,
                    prod_path=prod_path,
                    use_cloud_data_source=USE_CLOUD_DATA_SOURCE,
                ),
                'remote_annotation_path': fetch_annotation_signed_url(
                    encord_client,
                    prodname,
                ),
            }
            for db_id, prodname, prod_path in tqdm(
                products_to_download,
                desc='Resolving remote paths',
            )
        }

        # Optional sanity-check / logging
        missing = [
            k
            for k, v in job_dict.items()
            if v['remote_annotation_path'] is None
        ]
        if missing:
            logger.warning(
                f'WARNING: {len(missing)} product(s) have no remote annotation path: '
                f'{missing[:5]}{" …" if len(missing) > 5 else ""}'
            )

        processor = JobPullProcessor(
            dst=cfg.criteria.artifact_destination,
            db_conn=db_conn,
            q_success=load_attr(cfg.job_processor.query_success),
            q_failure=load_attr(cfg.job_processor.query_failure),
            encord_project=encord_project,
            allow_overwriting=cfg.job_processor.allow_overwriting,
            use_cloud_data_source=USE_CLOUD_DATA_SOURCE,
            aligner_method=cfg.job_processor.aligner_method,
            aligner_device=cfg.job_processor.aligner_device,
            save_only_annotation=cfg.job_processor.save_only_annotation,
            dry_run=cfg.job_processor.dry_run,
        )

        # In case of using remote bucket
        # the cloud bucket wrapper is initialized
        if USE_CLOUD_DATA_SOURCE:
            processor.set_cloud_bucket_wrapper(
                BucketWrapper(
                    project_name=str(env['DATA_ARCHIVE_GOOGLE_CLOUD_PROJECT']),
                    bucket_name=str(
                        env['DATA_ARCHIVE_GOOGLE_CLOUD_BUCKET_NAME']
                    ),
                )
            )

        # Loop processing over the products
        start = perf_counter()
        for idx, (
            db_id,
            job,
        ) in enumerate(tqdm(job_dict.items())):
            try:
                if dry_run:
                    logger.info(
                        f'[DRY-RUN] would process job id={db_id} : {job}'
                    )
                    continue
                processor(db_id, job)
            except Exception as e:
                if not dry_run:  # suppress UPDATE in dry run
                    logger.warning(
                        f'Exception {e} raised for id: {db_id} and {job}'
                    )
                    processor.stats.add('processing_failed', job['prodname'])
                    exec_query(
                        db_conn,
                        processor.q_failure,
                        params={
                            'exception_message': str(traceback.format_exc()),
                            'annotation_id': db_id,
                        },
                    )
        elapsed = perf_counter() - start
        time_msg = f'Total time: {elapsed:.1f} s  ({elapsed / len(job_dict):.2f} s/job)'
        logger.info(processor.stats.summary())
        stamp = datetime.now().strftime('%Y/%m/%d-%H:%M:%S')

        actual_usage_msg = usage_msg(
            use_cloud_data_source=processor.use_cloud_data_source,
            save_only_annotation=processor.save_only_annotation,
            destination_path=str(cfg.criteria.artifact_destination),
        )

        if not dry_run:
            debug_prefix = (
                '*[RUNNING IN DEBUG MODE]*\n'
                if cfg.debug_settings.limit_product_processed
                else ''
            )

            success_msg = (
                f'{debug_prefix}'
                '✅ Processing completed!\n\n'
                f'⏰ *{stamp}*\n\n'
                f'{actual_usage_msg}'
                f'🧐 Analyzed {len(job_dict)} products\n⌛ {time_msg}\n'
                f'👉 Correctly processed files {len(processor.stats.correctly_processed)}/{len(job_dict)}'
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
                num_lines = len(filebuf.getvalue().decode('utf-8').splitlines())
                upload_file_to_channel(
                    buffer=filebuf,
                    filename=filebuf.name,
                    token=str(env['SLACK_OAUTH']),
                    channel_id=str(env['SLACK_CHANNEL']),
                    initial_comment=f'{filebuf.name} -> {num_lines}',
                    thread_ts=thread_id,
                )

            mismatching_products = mismatched_products(
                cfg.criteria.artifact_destination
            )
            if len(mismatching_products) > 0:
                logger.warning(
                    f'Found {len(mismatching_products)} pair of annotation and product with different shapes!'
                )
                logger.warning(mismatching_products)
        db_conn.close()

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
