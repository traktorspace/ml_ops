import sys
import traceback
import warnings
from datetime import (
    datetime,
)
from pathlib import (
    Path,
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
from rasterio.errors import (
    NotGeoreferencedWarning,
)

from data_pipeline.utils.db_utils import exec_query, init_connection
from data_pipeline.utils.encord_utils import (
    fetch_annotations_duplicates,
    fetch_annotations_from_workflow_stages,
    fetch_client,
    fetch_project,
)
from data_pipeline.utils.hydra_helpers import (
    load_attr,
)
from data_pipeline.utils.slack_utils import (
    post_new_message_and_get_thread_id,
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

        encord_client = fetch_client(env['ENCORD_PRIVATE_KEY_PATH'])
        encord_project = fetch_project(
            client=encord_client,
            encord_project_hash=env['ENCORD_CLOUD_PROJECT_HASH'],
        )
        all_encord_annotations = fetch_annotations_from_workflow_stages(
            project=encord_project
        )

        if fetch_annotations_duplicates(all_encord_annotations):
            raise ValueError('Found duplicates in the annotations!')

        all_encord_annotations_unpacked = set().union(
            *all_encord_annotations.values()
        )

        all_archive_prods = set(
            Path(r[0]).name
            for r in exec_query(
                db_conn,
                load_attr(cfg.criteria.prod_selection_query),
            )
            if Path(r[0]).exists()
        )
        logger.info(len(all_encord_annotations_unpacked))
        logger.info(len(all_archive_prods))
        logger.info(len(all_archive_prods - all_encord_annotations_unpacked))
        post_new_message_and_get_thread_id(
            text=wrap_msg_with_project_name(
                msg=f'Found {len(all_archive_prods - all_encord_annotations_unpacked)} files on batman that could be uploaded on Encord',
                projname=projname,
            ),
            slack_bot_token=env['SLACK_OAUTH'],
            channel_id=env['SLACK_CHANNEL'],
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
            slack_bot_token=env['SLACK_OAUTH'],
            channel_id=env['SLACK_CHANNEL'],
        )
    finally:
        if db_conn is not None:
            try:
                db_conn.close()
            except Exception:
                logger.warning('Failed to close db connection!')


if __name__ == '__main__':
    main()
