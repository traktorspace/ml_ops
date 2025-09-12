import random
from pathlib import Path
from pprint import pprint

from dotenv import (
    dotenv_values,
)

from annotation_pipeline.utils.db_queries import (
    fetch_all_products_from_annotation_db,
    fetch_latest_approved_products,
)
from annotation_pipeline.utils.db_utils import (
    exec_query,
    init_connection,
)

env = dotenv_values(
    '/home/mlops/repos/ml_ops/annotation_pipeline/configs/cloud_annotation/.env'
)
db_conn = init_connection(env)
res = exec_query(
    db_conn,
    fetch_latest_approved_products,
    params={
        'min_cloud': 5,
        'max_cloud': 20,
        'created_after': '2025-08-01T00:00:00Z',
        'created_before': '2025-09-05T00:00:00Z',
    },
)

if res is None:
    raise RuntimeError('No results have been yielded from the query')

print(f'Found {len(res)} results')
pprint(random.choices(res, k=5))

all_fetched = [Path(r[0]).name for r in res]
print('Running general query to fetch all products')
all_ann = exec_query(
    db_conn,
    fetch_all_products_from_annotation_db,
)

if all_ann is None:
    raise RuntimeError('No annotations have been yielded from the query')

print(f'Found {len(all_ann)} annotations')
all_annotation_in_db = [Path(r[1]) for r in all_ann]

print(
    set(all_annotation_in_db).difference(set(all_fetched))
    == set(all_annotation_in_db)
)
