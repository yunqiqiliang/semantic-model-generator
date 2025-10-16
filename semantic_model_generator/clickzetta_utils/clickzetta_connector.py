from __future__ import annotations

import concurrent.futures
import re
from collections import defaultdict
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Dict, Generator, List, Optional, TypeVar, Union

import pandas as pd
from clickzetta.zettapark.session import Session
from loguru import logger

from semantic_model_generator.clickzetta_utils import env_vars
from semantic_model_generator.clickzetta_utils.utils import (
    clickzetta_connection,
    create_session,
)
from semantic_model_generator.data_processing.data_types import Column, Table

ConnectionType = TypeVar("ConnectionType", bound=Session)
AUTOGEN_TOKEN = "__"

_COMMENT_COL = "COMMENT"
_COLUMN_NAME_COL = "COLUMN_NAME"
_DATATYPE_COL = "DATA_TYPE"
_TABLE_SCHEMA_COL = "TABLE_SCHEMA"
_TABLE_NAME_COL = "TABLE_NAME"
_COLUMN_COMMENT_ALIAS = "COLUMN_COMMENT"
_TABLE_COMMENT_COL = "TABLE_COMMENT"
_IS_PRIMARY_KEY_COL = "IS_PRIMARY_KEY"

TIME_MEASURE_DATATYPES = [
    "DATE",
    "DATETIME",
    "TIMESTAMP",
    "TIMESTAMP_NTZ",
    "TIMESTAMP_LTZ",
    "TIMESTAMP_TZ",
    "TIME",
]
DIMENSION_DATATYPES = [
    "VARCHAR",
    "STRING",
    "TEXT",
    "CHAR",
    "CHARACTER",
    "NCHAR",
    "NVARCHAR",
    "BOOLEAN",
    "UUID",
]
MEASURE_DATATYPES = [
    "NUMBER",
    "DECIMAL",
    "NUMERIC",
    "INT",
    "INTEGER",
    "BIGINT",
    "SMALLINT",
    "TINYINT",
    "FLOAT",
    "DOUBLE",
    "REAL",
]
OBJECT_DATATYPES = [
    "VARIANT",
    "OBJECT",
    "ARRAY",
    "MAP",
    "JSON",
    "STRUCT",
    "GEOGRAPHY",
    "BINARY",
    "VARBINARY",
    "VECTOR",
    "VOID",
]


def _execute_query_to_pandas(connection: Any, query: str) -> pd.DataFrame:
    """
    Executes a SQL query and returns a pandas DataFrame while supporting both ClickZetta
    sessions and legacy connector shims.
    """

    logger.debug(f"Executing query: {query}")

    if hasattr(connection, "sql"):
        return connection.sql(query).to_pandas()

    cursor = connection.cursor()
    try:
        cursor.execute(query)
    except Exception:
        cursor.close()
        raise

    if hasattr(cursor, "fetch_pandas_all"):
        df = cursor.fetch_pandas_all()
    else:  # pragma: no cover - defensive branch
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        df = pd.DataFrame(rows, columns=columns)
    cursor.close()
    return df


class ClickzettaCursor:
    def __init__(self, session: Session):
        self._session = session
        self._df: Optional[pd.DataFrame] = None
        self.description: List[Any] = []

    def execute(self, query: str) -> "ClickzettaCursor":
        self._df = _execute_query_to_pandas(self._session, query)
        columns = [] if self._df is None else list(self._df.columns)
        self.description = [(col, None, None, None, None, None, None) for col in columns]
        return self

    def fetchone(self) -> Optional[tuple[Any, ...]]:
        if self._df is None or self._df.empty:
            return None
        return tuple(self._df.iloc[0].tolist())

    def fetchall(self) -> List[tuple[Any, ...]]:
        if self._df is None:
            return []
        return [tuple(row.tolist()) for _, row in self._df.iterrows()]

    def fetch_pandas_all(self) -> pd.DataFrame:
        if self._df is None:
            return pd.DataFrame()
        return self._df.copy()

    def close(self) -> None:
        self._df = None

    def __iter__(self):
        return iter(self.fetchall())


class ClickzettaConnectionProxy:
    def __init__(self, session: Session, config: Dict[str, str]):
        self.session = session
        self.config = config
        self.host = config.get("service", "")

    def cursor(self) -> ClickzettaCursor:
        return ClickzettaCursor(self.session)

    def close(self) -> None:
        self.session.close()


def _quote_identifier(name: str) -> str:
    return f'"{name}"'


def _qualify_table(workspace: str, schema_name: str, table_name: str) -> str:
    return ".".join(
        [_quote_identifier(workspace), _quote_identifier(schema_name), _quote_identifier(table_name)]
    )


def _value_is_true(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().strip('"').lower()
        return normalized == "true"
    return False


def _sanitize_identifier(value: Any, fallback: str = "") -> str:
    if value is None or value == "":
        return fallback
    normalized = str(value).strip()
    if normalized.startswith('"') and normalized.endswith('"') and len(normalized) >= 2:
        normalized = normalized[1:-1]
    return normalized


def _normalize_column_type(raw: Any) -> str:
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text.upper())
    for suffix in (" NOT NULL", " NULL"):
        if text.endswith(suffix):
            text = text[: -len(suffix)].strip()
    return text


def _get_table_comment(columns_df: pd.DataFrame) -> str:
    if columns_df.empty or _TABLE_COMMENT_COL not in columns_df:
        return ""
    table_comment = columns_df[_TABLE_COMMENT_COL].iloc[0]
    return str(table_comment) if table_comment else ""


def _get_column_comment(column_row: pd.Series) -> str:
    comment = column_row.get(_COLUMN_COMMENT_ALIAS)
    return str(comment) if comment else ""


def _fetch_distinct_values(
    session: Session,
    workspace: str,
    schema_name: str,
    table_name: str,
    column_name: str,
    ndv: int,
) -> Optional[List[str]]:
    workspace_part = _sanitize_identifier(workspace, workspace).upper() if workspace else ""
    schema_part = _sanitize_identifier(schema_name, schema_name).upper() if schema_name else ""
    table_part = _sanitize_identifier(table_name, table_name).upper()
    column_part = _sanitize_identifier(column_name, column_name).upper()

    qualified_parts = [part for part in (workspace_part, schema_part, table_part) if part]
    qualified_table = ".".join(qualified_parts)

    query = f"SELECT DISTINCT {column_part} FROM {qualified_table} LIMIT {ndv}"
    try:
        df = session.sql(query).to_pandas()
        if df.empty:
            return None
        first_col = df.columns[0]
        return [str(value) for value in df[first_col].tolist()]
    except Exception as exc:  # pragma: no cover - logging defensive path
        logger.warning(
            "Failed to sample values for {}.{}.{}.{}: {}",
            workspace,
            schema_name,
            table_name,
            column_name,
            exc,
        )
        return None


def _get_column_representation(
    session: Session,
    workspace: str,
    schema_name: str,
    table_name: str,
    column_row: pd.Series,
    column_index: int,
    ndv: int,
) -> Column:
    column_name = column_row[_COLUMN_NAME_COL]
    column_datatype_raw = column_row[_DATATYPE_COL]
    if isinstance(column_datatype_raw, str):
        column_datatype = column_datatype_raw
    else:
        column_datatype = str(column_datatype_raw)
    column_datatype = _normalize_column_type(column_datatype)
    normalized_type = column_datatype.split("(")[0].strip()
    column_values = (
        _fetch_distinct_values(
            session=session,
            workspace=workspace,
            schema_name=schema_name,
            table_name=table_name,
            column_name=column_name,
            ndv=ndv,
        )
        if ndv > 0
        else None
    )

    raw_primary = column_row.get(_IS_PRIMARY_KEY_COL)
    if isinstance(raw_primary, str):
        normalized = raw_primary.strip().upper()
        is_primary = normalized in {"TRUE", "YES", "1"}
    else:
        is_primary = bool(raw_primary)

    return Column(
        id_=column_index,
        column_name=column_name,
        comment=_get_column_comment(column_row),
        column_type=column_datatype,
        values=column_values,
        is_primary_key=is_primary,
    )


def get_table_representation(
    session: Session,
    workspace: str,
    schema_name: str,
    table_name: str,
    table_index: int,
    ndv_per_column: int,
    columns_df: pd.DataFrame,
    max_workers: int,
) -> Table:
    table_comment = _get_table_comment(columns_df)

    def _get_col(col_index: int, column_row: pd.Series) -> Column:
        return _get_column_representation(
            session=session,
            workspace=workspace,
            schema_name=schema_name,
            table_name=table_name,
            column_row=column_row,
            column_index=col_index,
            ndv=ndv_per_column,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(_get_col, idx, row): idx
            for idx, (_, row) in enumerate(columns_df.iterrows())
        }
        ordered_columns: Dict[int, Column] = {}
        for future in concurrent.futures.as_completed(future_to_index):
            ordered_columns[future_to_index[future]] = future.result()
        columns = [ordered_columns[i] for i in sorted(ordered_columns)]

    return Table(
        id_=table_index,
        name=table_name,
        comment=table_comment,
        columns=columns,
    )


_CATALOG_CATEGORY_CACHE: Dict[str, str] = {}


def _catalog_category(session: Session, workspace: str) -> str:
    workspace_upper = workspace.upper()
    if not workspace_upper:
        return "UNKNOWN"
    cached = _CATALOG_CATEGORY_CACHE.get(workspace_upper)
    if cached:
        return cached
    try:
        df = session.sql("SHOW CATALOGS").to_pandas()
    except Exception as exc:  # pragma: no cover
        logger.debug("SHOW CATALOGS failed: {}", exc)
        _CATALOG_CATEGORY_CACHE[workspace_upper] = "UNKNOWN"
        return "UNKNOWN"

    if df.empty:
        _CATALOG_CATEGORY_CACHE[workspace_upper] = "UNKNOWN"
        return "UNKNOWN"

    df.columns = [str(col).upper() for col in df.columns]
    name_col = next((col for col in ("WORKSPACE_NAME", "NAME", "CATALOG_NAME") if col in df.columns), None)
    category_col = next((col for col in ("CATEGORY",) if col in df.columns), None)
    if not name_col or not category_col:
        _CATALOG_CATEGORY_CACHE[workspace_upper] = "UNKNOWN"
        return "UNKNOWN"

    for _, row in df.iterrows():
        name = str(row[name_col]).upper()
        if name == workspace_upper:
            category = str(row[category_col]).upper()
            _CATALOG_CATEGORY_CACHE[workspace_upper] = category
            return category

    _CATALOG_CATEGORY_CACHE[workspace_upper] = "UNKNOWN"
    return "UNKNOWN"


def get_table_primary_keys(
    session: Session,
    workspace: str,
    schema_name: str,
    table_name: str,
) -> Optional[List[str]]:
    catalog = workspace.upper()
    schema = schema_name.upper()
    table = table_name.upper()

    def _run(query: str) -> Optional[List[str]]:
        df = session.sql(query).to_pandas()
        if df.empty:
            return None
        return [str(value) for value in df.iloc[:, 0].tolist()]

    base_condition = [
        "tc.constraint_type = 'PRIMARY KEY'",
        f"upper(tc.table_schema) = '{schema}'",
        f"upper(tc.table_name) = '{table}'",
    ]
    if catalog:
        base_condition.append(f"upper(tc.table_catalog) = '{catalog}'")
    where_sql = " AND ".join(base_condition)

    sys_query = f"""
SELECT kc.column_name
FROM sys.information_schema.table_constraints tc
JOIN sys.information_schema.key_column_usage kc
  ON tc.constraint_catalog = kc.constraint_catalog
 AND tc.constraint_schema = kc.constraint_schema
 AND tc.constraint_name = kc.constraint_name
WHERE {where_sql}
ORDER BY kc.ordinal_position
"""
    try:
        result = _run(sys_query)
        if result is not None:
            return result
    except Exception:
        logger.debug("Primary key lookup via sys.information_schema failed; falling back.")

    fallback_query = f"""
SELECT kc.column_name
FROM information_schema.table_constraints tc
JOIN information_schema.key_column_usage kc
  ON tc.constraint_name = kc.constraint_name
WHERE {where_sql}
ORDER BY kc.ordinal_position
"""
    try:
        result = _run(fallback_query)
        if result is not None:
            return result
    except Exception as exc:
        logger.warning("Primary key lookup failed for {}.{}.{}: {}", workspace, schema_name, table_name, exc)
    return None


def _build_information_schema_query(
    workspace: str,
    table_schema: Optional[str],
    table_names: Optional[List[str]],
) -> str:
    where_conditions: List[str] = [
        "1=1"
    ]
    if table_schema:
        where_conditions.append(f"upper(t.table_schema) = '{table_schema.upper()}'")
    if table_names:
        formatted_names = ", ".join(f"'{name.upper()}'" for name in table_names)
        where_conditions.append(f"upper(t.table_name) IN ({formatted_names})")

    where_clause = " AND ".join(where_conditions)
    base = "information_schema"
    return f"""
SELECT
    t.table_schema AS {_TABLE_SCHEMA_COL},
    t.table_name AS {_TABLE_NAME_COL},
    c.column_name AS {_COLUMN_NAME_COL},
    c.data_type AS {_DATATYPE_COL},
    c.comment AS {_COLUMN_COMMENT_ALIAS},
    t.comment AS {_TABLE_COMMENT_COL},
    c.is_primary_key AS {_IS_PRIMARY_KEY_COL}
FROM information_schema.tables t
JOIN information_schema.columns c
  ON t.table_schema = c.table_schema
 AND t.table_name = c.table_name
WHERE {where_clause}
"""


def _fetch_columns_via_show(
    session: Session,
    workspace: str,
    table_schema: Optional[str],
    table_names: Optional[List[str]],
) -> pd.DataFrame:
    if not table_names:
        return pd.DataFrame()

    rows: List[pd.DataFrame] = []
    catalog = workspace.upper()
    schema = table_schema.upper() if table_schema else ""

    for table_name in table_names:
        qualified_parts = [part for part in (catalog, schema, table_name.upper()) if part]
        qualified_table = ".".join(qualified_parts)
        query = f"SHOW COLUMNS IN {qualified_table}"
        try:
            df = session.sql(query).to_pandas()
        except Exception as exc:
            logger.debug("SHOW COLUMNS fallback failed for {}: {}", qualified_table, exc)
            continue
        if df.empty:
            continue
        df.columns = [str(col).upper() for col in df.columns]
        schema_col = next((col for col in ("TABLE_SCHEMA", "SCHEMA_NAME") if col in df.columns), None)
        table_col = next((col for col in ("TABLE_NAME", "NAME") if col in df.columns), None)
        column_col = next((col for col in ("COLUMN_NAME", "NAME") if col in df.columns and col != table_col), None)
        datatype_col = next((col for col in ("DATA_TYPE", "TYPE") if col in df.columns), None)
        comment_col = next((col for col in ("COMMENT", "COLUMN_COMMENT") if col in df.columns), None)

        normalized = pd.DataFrame()
        normalized[_TABLE_SCHEMA_COL] = df[schema_col] if schema_col else table_schema
        normalized[_TABLE_NAME_COL] = df[table_col] if table_col else table_name
        normalized[_COLUMN_NAME_COL] = df[column_col] if column_col else df.index.astype(str)
        normalized[_DATATYPE_COL] = df[datatype_col] if datatype_col else ""
        normalized[_COLUMN_COMMENT_ALIAS] = df[comment_col] if comment_col else ""
        normalized[_TABLE_COMMENT_COL] = ""
        normalized[_IS_PRIMARY_KEY_COL] = False
        rows.append(normalized)

    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    combined[_TABLE_SCHEMA_COL] = combined[_TABLE_SCHEMA_COL].astype(str).str.upper()
    combined[_TABLE_NAME_COL] = combined[_TABLE_NAME_COL].astype(str).str.upper()
    combined[_COLUMN_NAME_COL] = combined[_COLUMN_NAME_COL].astype(str)
    combined.columns = [str(col).upper() for col in combined.columns]
    return combined


def get_valid_schemas_tables_columns_df(
    session: Session,
    workspace: str,
    table_schema: Optional[str] = None,
    table_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    category = _catalog_category(session, workspace)
    skip_information_schema = category in {"SHARED", "EXTERNAL"}

    result = pd.DataFrame()
    if not skip_information_schema:
        query = _build_information_schema_query(
            workspace=workspace,
            table_schema=table_schema,
            table_names=table_names,
        )
        logger.debug("Running metadata query:\n{}", query)
        try:
            result = session.sql(query).to_pandas()
        except Exception as exc:
            logger.debug("information_schema query failed: {}", exc)
            result = pd.DataFrame()

    if result.empty:
        result = _fetch_columns_via_show(
            session=session,
            workspace=workspace,
            table_schema=table_schema,
            table_names=table_names,
        )

    if result.empty:
        return result

    # Normalize column labels for downstream operations.
    result.columns = [str(col).upper() for col in result.columns]
    if _TABLE_NAME_COL in result.columns:
        result[_TABLE_NAME_COL] = result[_TABLE_NAME_COL].astype(str).str.upper()
    if _TABLE_SCHEMA_COL in result.columns:
        result[_TABLE_SCHEMA_COL] = result[_TABLE_SCHEMA_COL].astype(str).str.upper()
    if _IS_PRIMARY_KEY_COL in result.columns:
        def _normalize_pk(value: Any) -> bool:
            if isinstance(value, bool):
                return value
            if value is None:
                return False
            normalized = str(value).strip().upper()
            return normalized in {"TRUE", "YES", "1"}

        result[_IS_PRIMARY_KEY_COL] = result[_IS_PRIMARY_KEY_COL].apply(_normalize_pk)
    return result


def fetch_databases(session: Session) -> List[str]:
    df = session.sql("SHOW CATALOGS").to_pandas()
    for candidate in ("workspace_name", "name", "catalog_name"):
        if candidate in df.columns:
            return [str(value) for value in df[candidate].tolist()]
    if not df.empty:
        return [str(value) for value in df.iloc[:, 0].tolist()]
    return []


def fetch_warehouses(session: Session) -> List[str]:
    df = session.sql("SHOW VCLUSTERS").to_pandas()
    for candidate in ("name", "vcluster_name"):
        if candidate in df.columns:
            return [str(value) for value in df[candidate].tolist()]
    if not df.empty:
        return [str(value) for value in df.iloc[:, 0].tolist()]
    return []


def fetch_schemas_in_database(session: Session, workspace: str) -> List[str]:
    temp_session = create_session(
        service=env_vars.CLICKZETTA_SERVICE or "",
        instance=env_vars.CLICKZETTA_INSTANCE or "",
        workspace=workspace,
        schema="",
        username=env_vars.CLICKZETTA_USERNAME or "",
        password=env_vars.CLICKZETTA_PASSWORD or "",
        vcluster=env_vars.CLICKZETTA_VCLUSTER or "default_ap",
    )
    try:
        df = temp_session.sql("SHOW SCHEMAS").to_pandas()
    finally:
        temp_session.close()

    for candidate in ("name", "schema_name"):
        if candidate in df.columns:
            return [f"{workspace}.{value}" for value in df[candidate].tolist()]
    if not df.empty:
        return [f"{workspace}.{value}" for value in df.iloc[:, 0].tolist()]
    return []


def fetch_tables_views_in_schema(
    session: Session,
    schema_name: str,
) -> List[str]:
    parts = schema_name.split(".", maxsplit=1)
    workspace = parts[0]
    schema = parts[1] if len(parts) > 1 else ""
    workspace_upper = workspace.upper()
    schema_upper = schema.upper()

    target = ""
    try:
        if workspace_upper and schema_upper:
            df = session.sql(f"SHOW TABLES IN {workspace_upper}.{schema_upper}").to_pandas()
        else:
            df = session.sql("SHOW TABLES").to_pandas()
    except Exception as exc:  # pragma: no cover
        logger.warning("Failed to list tables for {}: {}", schema_name, exc)
        return []

    if df.empty:
        return []

    # Normalize column names but keep the original values
    df.columns = [str(col).upper() for col in df.columns]
    name_column = "TABLE_NAME" if "TABLE_NAME" in df.columns else df.columns[0]
    schema_column = next(
        (col for col in ("SCHEMA_NAME", "TABLE_SCHEMA", "NAMESPACE") if col in df.columns),
        None,
    )
    catalog_column = next(
        (col for col in ("CATALOG_NAME", "WORKSPACE_NAME", "TABLE_CATALOG") if col in df.columns),
        None,
    )

    results: List[str] = []
    for _, row in df.iterrows():
        if _value_is_true(row.get("IS_VIEW")) and not _value_is_true(row.get("IS_MATERIALIZED_VIEW")):
            continue
        if _value_is_true(row.get("IS_EXTERNAL")):
            continue
        # keep materialized views
        catalog_part = _sanitize_identifier(
            row[catalog_column] if catalog_column else workspace, workspace
        )
        schema_part = _sanitize_identifier(
            row[schema_column] if schema_column else schema, schema
        )
        table_part = _sanitize_identifier(row[name_column], "")
        if not table_part:
            continue
        parts_fqn = [catalog_part] if catalog_part else []
        if schema_part:
            parts_fqn.append(schema_part)
        parts_fqn.append(table_part)
        fully_qualified = ".".join(parts_fqn)
        results.append(fully_qualified)

    seen: set[str] = set()
    ordered: List[str] = []
    for name in results:
        if name not in seen:
            ordered.append(name)
            seen.add(name)
    return ordered


def fetch_stages_in_schema(connection: Any, schema_name: str) -> List[str]:
    """
    Returns the volumes (ClickZetta) or legacy stages available within a schema.
    """

    if "." in schema_name:
        workspace, schema = schema_name.split(".", 1)
    else:
        workspace, schema = schema_name, ""

    queries: List[str] = []
    if schema:
        queries.append(f"SHOW VOLUMES IN {workspace}.{schema}")
        queries.append(f"SHOW STAGES IN SCHEMA {workspace}.{schema}")
    else:
        queries.append(f"SHOW VOLUMES IN {workspace}")
        queries.append(f"SHOW STAGES IN DATABASE {workspace}")

    stage_names: List[str] = ["volume:user://~/semantic_models/"]
    seen: set[str] = set(stage_names)

    df = pd.DataFrame()
    last_error: Optional[Exception] = None
    for query in queries:
        try:
            df = _execute_query_to_pandas(connection, query)
            if not df.empty:
                break
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Stage/volume query failed ({}): {}", query, exc)
            last_error = exc
    else:
        if last_error:
            raise last_error

    if df.empty:
        return stage_names

    name_column: Optional[str] = None
    for column in df.columns:
        if column.lower() in {"name", "volume_name", "stage_name"}:
            name_column = column
            break
    if not name_column:
        name_column = df.columns[0]

    for value in df[name_column].tolist():
        name = str(value)
        if name not in seen:
            stage_names.append(name)
            seen.add(name)
    return stage_names


def fetch_yaml_names_in_stage(
    connection: Any, stage: str, include_yml: bool = False
) -> List[str]:
    """
    Lists YAML files stored inside a ClickZetta volume (or legacy stage).
    """

    stage = stage.strip()

    def _filter_yaml_names(values: List[str], base_prefix: str = "") -> List[str]:
        filtered: List[str] = []
        for value in values:
            normalized = value.strip()
            if base_prefix and normalized.startswith(base_prefix):
                normalized = normalized[len(base_prefix) :]
            normalized = normalized.lstrip("/")
            lower = normalized.lower()
            if include_yml:
                if lower.endswith(".yaml") or lower.endswith(".yml"):
                    filtered.append(normalized)
            else:
                if lower.endswith(".yaml"):
                    filtered.append(normalized)
        return filtered

    if stage.lower().startswith("volume:user://"):
        volume_body = stage[len("volume:") :]
        # Normalize relative directory
        relative = volume_body[len("user://") :] if volume_body.startswith("user://") else volume_body
        relative = relative.lstrip("~/")
        relative = relative.strip("/")

        list_sql = "LIST USER VOLUME"
        if relative:
            list_sql += f" SUBDIRECTORY '{relative}/'"

        try:
            df = _execute_query_to_pandas(connection, list_sql)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Failed to LIST user volume for {}: {}", stage, exc)
            return []
        if df.empty:
            return []

        column_name: Optional[str] = None
        for candidate in ("name", "relative_path", "path"):
            if candidate in df.columns:
                column_name = candidate
                break
        if not column_name:
            column_name = df.columns[0]

        values = [str(value) for value in df[column_name].tolist() if value]
        base_prefix = f"{relative}/" if relative else ""
        deduped: List[str] = []
        seen_names: set[str] = set()
        for name in _filter_yaml_names(values, base_prefix=base_prefix):
            if name not in seen_names:
                deduped.append(name)
                seen_names.add(name)
        return deduped

    # Legacy stage flow (compatibility shim)
    stage_candidates: List[str] = []
    cleaned = stage.lstrip("@")
    if cleaned:
        stage_candidates.append(cleaned)
    if "." in cleaned:
        stage_candidates.append(cleaned.split(".")[-1])

    results: List[str] = []
    seen: set[str] = set()
    for candidate in stage_candidates:
        list_sql = f"LIST @{candidate}"
        try:
            df = _execute_query_to_pandas(connection, list_sql)
        except Exception as exc:  # pragma: no cover - defensive fallback
            logger.debug("Failed to LIST contents for {}: {}", candidate, exc)
            continue
        if df.empty:
            continue
        name_column = "name" if "name" in df.columns else df.columns[0]
        values = [str(value) for value in df[name_column].tolist() if value]
        for file_name in _filter_yaml_names(values):
            if file_name not in seen:
                results.append(file_name)
                seen.add(file_name)
        if results:
            break
    return results


def fetch_table_schema(session: Session, table_fqn: str) -> Dict[str, str]:
    try:
        df = session.sql(f"DESCRIBE TABLE {table_fqn}").to_pandas()
    except Exception as exc:
        logger.error("Unable to describe table {}: {}", table_fqn, exc)
        raise
    schema: Dict[str, str] = {}
    for _, row in df.iterrows():
        if row["kind"].upper() == "COLUMN":
            schema[str(row["name"])] = str(row["type"])
    return schema


def fetch_table(session: Session, table_fqn: str) -> pd.DataFrame:
    return session.sql(f"SELECT * FROM {table_fqn}").to_pandas()


def create_table_in_schema(
    session: Session,
    table_fqn: str,
    columns_schema: Dict[str, str],
) -> bool:
    fields = ", ".join(f"{_quote_identifier(name)} {dtype}" for name, dtype in columns_schema.items())
    query = f"CREATE TABLE IF NOT EXISTS {table_fqn} ({fields})"
    try:
        session.sql(query).collect()
        return True
    except Exception as exc:
        logger.error("Error creating table {}: {}", table_fqn, exc)
        return False


def get_table_hash(session: Session, table_fqn: str) -> str:
    df = session.sql(f"SELECT * FROM {table_fqn}").to_pandas()
    if df.empty:
        return "EMPTY"
    series = pd.util.hash_pandas_object(df, index=True)
    total = int(series.sum() % (1 << 64))
    return f"{total:016x}"


def execute_query(session: Session, query: str) -> Union[pd.DataFrame, str]:
    if not query:
        raise ValueError("Query string is empty")
    try:
        return session.sql(query).to_pandas()
    except Exception as exc:
        logger.info("Query execution failed: {}", exc)
        return str(exc)


class ClickzettaConnector:
    def __init__(
        self,
        *,
        max_workers: int = 1,
        overrides: Optional[Dict[str, str]] = None,
        hints: Optional[Dict[str, str]] = None,
    ):
        self._max_workers = max_workers
        self._base_config = env_vars.build_base_connection_config()
        if overrides:
            self._base_config.update({k: v for k, v in overrides.items() if v})
        self._hints = hints or env_vars.CLICKZETTA_HINTS.copy()

    def _require(self, key: str) -> str:
        value = self._base_config.get(key, "")
        if not value:
            raise ValueError(f"Missing ClickZetta configuration value for {key}")
        return value

    @contextmanager
    def connect(
        self,
        workspace: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> Generator[Session, None, None]:
        session = create_session(
            service=self._require("service"),
            instance=self._require("instance"),
            workspace=workspace or self._require("workspace"),
            schema=schema_name or self._require("schema"),
            username=self._require("username"),
            password=self._require("password"),
            vcluster=self._base_config.get("vcluster", "default_ap"),
            hints=self._hints,
        )
        try:
            yield session
        finally:
            session.close()

    def open_session(
        self,
        workspace: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> Session:
        return create_session(
            service=self._require("service"),
            instance=self._require("instance"),
            workspace=workspace or self._require("workspace"),
            schema=schema_name or self._require("schema"),
            username=self._require("username"),
            password=self._require("password"),
            vcluster=self._base_config.get("vcluster", "default_ap"),
            hints=self._hints,
        )

    def execute(
        self,
        session: Session,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> pd.DataFrame:
        if params:
            for key, value in params.items():
                placeholder = f":{key}"
                query = query.replace(placeholder, str(value))
        return session.sql(query).to_pandas()

    def fetch_table_details(
        self,
        workspace: str,
        schema_name: str,
        table_names: List[str],
        ndv_per_column: int,
    ) -> List[Table]:
        with self.connect(workspace=workspace, schema_name=schema_name) as session:
            metadata = get_valid_schemas_tables_columns_df(
                session=session,
                workspace=workspace,
                table_schema=schema_name,
                table_names=table_names,
            )
            grouped = defaultdict(pd.DataFrame)
            for table_name, group in metadata.groupby(_TABLE_NAME_COL):
                grouped[str(table_name).upper()] = group
            tables: List[Table] = []
            for index, table_name in enumerate(table_names):
                columns_df = grouped.get(table_name.upper())
                if columns_df is None or columns_df.empty:
                    continue
                table = get_table_representation(
                    session=session,
                    workspace=workspace,
                    schema_name=schema_name,
                    table_name=table_name,
                    table_index=index,
                    ndv_per_column=ndv_per_column,
                    columns_df=columns_df,
                    max_workers=self._max_workers,
                )
                tables.append(table)
            return tables

    def open_connection(
        self,
        workspace: Optional[str] = None,
        schema_name: Optional[str] = None,
    ) -> ClickzettaConnectionProxy:
        session = self.open_session(workspace=workspace, schema_name=schema_name)
        return ClickzettaConnectionProxy(session, self._base_config.copy())
