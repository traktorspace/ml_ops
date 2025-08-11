from psycopg.sql import SQL

# Fetch the most recent path in case of duplicates
fetch_unique_products_paths = SQL("""
SELECT sub.s3_path
FROM (
        SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY s3_path
                    ORDER BY created_on DESC
                ) AS rn
        FROM prod.l1ab_product AS lap
        {where_clause}
) sub
WHERE rn = 1
""")

# fetch_unique_products_paths = SQL("""
# SELECT sub.s3_path
# FROM (
#         SELECT *,
#                 ROW_NUMBER() OVER (
#                     PARTITION BY s3_path
#                     ORDER BY created_on DESC
#                 ) AS rn
#         FROM prod.l1ab_product AS lap
#         {where_clause}
# ) sub
# WHERE rn = 1
# """)

# Fetch all products
fetch_all_products_paths = SQL("""
SELECT s3_path
FROM prod.l1ab_product
""")

fetch_all_products_paths_that_passed_qc = SQL("""
SELECT s3_path
FROM prod.l1ab_product
WHERE quality_approved = TRUE
""")

fetch_all_products_that_passed_qc = SQL("""
SELECT id, s3_path
FROM prod.l1ab_product
WHERE quality_approved = TRUE
""")

insert_cloud_annotation_row = SQL(""" 
INSERT INTO annotations.clouds_annotations (product_name, created_on, bucket_path, uploaded_on_encord, annotated_and_downloaded, contains_snow, product_id)
VALUES (%s, %s, %s, %s, %s, %s, %s)
""")

delete_cloud_annotation_row = SQL("""
DELETE FROM annotations.clouds_annotations
WHERE id = %s
""")

fetch_all_products_from_annotation_db = SQL("""
SELECT id, product_name
FROM annotations.clouds_annotations
""")

fetch_encord_uploaded_but_not_downloaded_products_from_annotation_db = SQL("""
SELECT
  cloud_ann.id,
  cloud_ann.product_name,
  lap.s3_path
FROM annotations.clouds_annotations AS cloud_ann 
JOIN prod.l1ab_product AS lap
  ON cloud_ann.product_id = lap.id
WHERE uploaded_on_encord = TRUE
and annotated_and_downloaded = FALSE
""")

fetch_all_encord_uploaded_prods_from_annotation_db = SQL("""
SELECT id, product_name
FROM annotations.clouds_annotations
WHERE uploaded_on_encord = TRUE
""")

fetch_join_annotation_and_prod_tables = SQL("""
SELECT
  cloud_ann.id,
  cloud_ann.product_name,
  lap.s3_path
FROM annotations.clouds_annotations AS cloud_ann 
JOIN prod.l1ab_product AS lap
  ON cloud_ann.product_id = lap.id
{where_clause}
""")

write_exception = SQL("""
UPDATE annotations.clouds_annotations
SET    exception_raised = %(exception_message)s
WHERE  id               = %(annotation_id)s
""")

mark_annotation_processed = SQL("""
UPDATE annotations.clouds_annotations
SET    annotated_and_downloaded = TRUE
WHERE  id               = %(annotation_id)s
""")
