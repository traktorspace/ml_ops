from typing import Any, Mapping, Sequence

import psycopg
from psycopg import sql

ExecuteParams = Sequence[Any] | Mapping[str, Any]


def build_db_connection_string(
    host: str, port: int, db_name: str, db_user: str, db_password: str
):
    """Build and return a connection string for connecting to a postgres DB."""
    conn_str = f'host={host} port={int(port)} dbname={db_name} user={db_user} password={db_password}'
    return conn_str


def init_connection(config: dict):
    """
    Open a new PostgreSQL connection using the values found in *config*.

    The dictionary must contain the following keys (all strings unless noted
    otherwise):

    - ``POSTGRES_DB_HOST`` - database host name or IP address
    - ``POSTGRES_DB_PORT`` - port number (int or str)
    - ``POSTGRES_DB_NAME`` - database name
    - ``POSTGRES_DB_USER`` - database user
    - ``POSTGRES_DB_PASS`` - user password

    Parameters
    ----------
    config
        Mapping that provides the credentials and network parameters listed
        above.

    Returns
    -------
    psycopg.Connection
        An **open** psycopg (v3) connection.  The caller is responsible for
        closing it (or returning it to a pool).
    """
    conn_str = build_db_connection_string(
        host=config['POSTGRES_DB_HOST'],
        port=config['POSTGRES_DB_PORT'],
        db_name=config['POSTGRES_DB_NAME'],
        db_user=config['POSTGRES_DB_USER'],
        db_password=config['POSTGRES_DB_PASS'],
    )

    return psycopg.connect(conninfo=conn_str)


def exec_query(
    connection: psycopg.Connection,
    query: str | sql.Composable | sql.SQL,
    params: ExecuteParams | None = (),
    fetch: bool = True,
):
    """
    Execute a SQL statement and return all rows.

    Opens a new PostgreSQL connection using the environment variables
    `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, and `DB_PASS`.
    The connection is closed in a ``finally`` block.

    Parameters
    ----------
    connection: an already instantiated psycopg Connection
    query:
        Parametrised SQL query containing ``%s`` placeholders.
    params:
        Values to bind to the placeholders in *query*.
        Defaults to an empty tuple.
    fetch:
        If ``True`` (default) the function will try to return
        ``cursor.fetchall()`` when the statement yields a result set.
        Set to ``False`` for INSERT/UPDATE/DELETE or DDL.


    Returns
    -------
    fetch:
        If ``True`` (default) the function will try to return
        ``cursor.fetchall()`` when the statement yields a result set.
        Set to ``False`` for INSERT/UPDATE/DELETE or DDL.

    """
    try:
        with connection.cursor() as cur:
            cur.execute(query, params)  # type: ignore
            # fetch only when requested and there is something to fetch
            if fetch and cur.description is not None:
                return cur.fetchall()  # ret rows

            connection.commit()  # persist INSERT/UPDATE/DELETE
            return None
    except Exception as e:
        connection.rollback()
        raise e


def build_query(
    base_query: sql.Composed | sql.SQL, new_clause: sql.Composed | None = None
) -> sql.Composed | sql.SQL:
    """
    Compose or extend a SQL query with a WHERE clause.

    If the base query already includes a WHERE clause, appends with AND.
    Otherwise, adds a new WHERE clause.

    Parameters
    ----------
    base_query : sql.Composed or sql.SQL
        Base SQL query.
    new_clause : sql.Composed or None
        Additional WHERE clause fragment (without the 'WHERE' keyword).

    Returns
    -------
    sql.Composed
        The final query with the new clause integrated.

    Example
    -------
    >>> from psycopg import sql
    >>> base = sql.SQL("SELECT * FROM my_table")
    >>> clause = sql.SQL("s3_path LIKE %s")
    >>> final_query = build_query(base, clause)
    >>> print(final_query.as_string(conn))
    SELECT * FROM my_table WHERE s3_path LIKE %s

    >>> base_with_where = sql.SQL("SELECT * FROM my_table WHERE is_active = TRUE")
    >>> final_query = build_query(base_with_where, clause)
    >>> print(final_query.as_string(conn))
    SELECT * FROM my_table WHERE is_active = TRUE AND s3_path LIKE %s
    """
    if not new_clause:
        return base_query

    # Convert to string to check if it already includes a WHERE clause
    query_str = str(base_query).upper()

    if 'WHERE' in query_str:
        return base_query + sql.SQL(' AND ') + new_clause
    else:
        return base_query + sql.SQL(' WHERE ') + new_clause


# def build_like_filter(column: str, values: list[str]) -> tuple[str, tuple]:
#     """
#     Build a WHERE fragment such as
#     "s3_path LIKE %s OR s3_path LIKE %s …"  plus its params tuple.
#     Each input value is wrapped with % for a substring match.

#     Example
#     -------
#     prods_filter, params = build_like_filter('s3_path',folders)
#     result = exec_query(build_query(Q.fetch_unique_products_paths, prods_filter), params)
#     """
#     # if not values:
#     #     return "", ()
#     # clause = " OR ".join(f"{column} LIKE %s" for _ in values)
#     # params = tuple(f"%{v}%" for v in values)
#     # return clause, params

#     if not values:
#         return sql.SQL(''), ()

#     col_id = sql.Identifier(column)
#     likes = sql.SQL(' OR ').join(
#         sql.SQL('{} LIKE {}').format(col_id, sql.Placeholder()) for _ in values
#     )
#     params = tuple(f'%{v}%' for v in values)
#     return likes, params


def build_like_filter(
    column: str, values: list[str]
) -> tuple[sql.Composed | sql.SQL, tuple]:
    """
    Build a LIKE clause with placeholders and return the SQL and parameters.

    Parameters
    ----------
    column :
        Column name for LIKE filtering.
    values :
        Values for LIKE filtering.

    Returns
    -------
    tuple[sql.Composed, tuple]
        SQL WHERE clause (without 'WHERE') and parameter values.
    """
    if not values:
        return sql.SQL(''), ()

    col_id = sql.Identifier(column)
    clause = sql.SQL(' OR ').join(
        sql.SQL('{} LIKE {}').format(col_id, sql.Placeholder()) for _ in values
    )
    params = tuple(f'%{v}%' for v in values)
    return clause, params
