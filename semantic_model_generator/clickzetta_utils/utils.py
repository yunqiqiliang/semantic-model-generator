from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable

from clickzetta.zettapark.session import Session

from semantic_model_generator.data_processing.data_types import FQNParts

DEFAULT_HINTS: Dict[str, str] = {
    "sdk.job.timeout": "300",
    "query_tag": "semantic-model-generator",
    "cz.storage.parquet.vector.index.read.memory.cache": "true",
    "cz.storage.parquet.vector.index.read.local.cache": "false",
    "cz.sql.table.scan.push.down.filter": "true",
    "cz.sql.table.scan.enable.ensure.filter": "true",
    "cz.storage.always.prefetch.internal": "true",
    "cz.optimizer.generate.columns.always.valid": "true",
    "cz.sql.index.prewhere.enabled": "true",
    "cz.storage.parquet.enable.io.prefetch": "false",
}


def create_fqn_table(fqn_str: str) -> FQNParts:
    """
    Splits a fully qualified table name into its ClickZetta components.

    Expected format: ``{workspace}.{schema}.{table}``.
    """

    if fqn_str.count(".") != 2:
        raise ValueError(
            "Expected a fully-qualified identifier in the form "
            "{workspace}.{schema}.{table}; "
            f"received {fqn_str!r}"
        )
    workspace, schema, table = fqn_str.split(".")
    return FQNParts(
        database=workspace.upper(), schema_name=schema.upper(), table=table.upper()
    )


def _build_session_config(
    *,
    service: str,
    instance: str,
    workspace: str,
    schema: str,
    username: str,
    password: str,
    vcluster: str,
    hints: Dict[str, str] | None = None,
) -> Dict[str, object]:
    config: Dict[str, object] = {
        "service": service,
        "instance": instance,
        "workspace": workspace,
        "schema": schema,
        "username": username,
        "password": password,
        "vcluster": vcluster,
    }
    merged_hints = dict(DEFAULT_HINTS)
    if hints:
        merged_hints.update({k: str(v) for k, v in hints.items()})
    config["hints"] = merged_hints
    return config


def _apply_session_context(session: Session, *, schema: str, vcluster: str) -> None:
    for component, value in _iter_non_empty(
        ("schema", schema),
        ("vcluster", vcluster),
    ):
        session.sql(f"USE {component.upper()} {value.upper()}")


def _iter_non_empty(*pairs: tuple[str, str]) -> Iterable[tuple[str, str]]:
    for key, value in pairs:
        if value:
            yield key, value


def create_session(
    *,
    service: str,
    instance: str,
    workspace: str,
    schema: str,
    username: str,
    password: str,
    vcluster: str,
    hints: Dict[str, str] | None = None,
) -> Session:
    """
    Creates a ClickZetta Session pre-configured with workspace/schema context.
    """

    session = Session.builder.configs(
        _build_session_config(
            service=service,
            instance=instance,
            workspace=workspace,
            schema=schema,
            username=username,
            password=password,
            vcluster=vcluster,
            hints=hints,
        )
    ).create()
    _apply_session_context(session, schema=schema, vcluster=vcluster)
    return session


@contextmanager
def clickzetta_connection(
    *,
    service: str,
    instance: str,
    workspace: str,
    schema: str,
    username: str,
    password: str,
    vcluster: str,
    hints: Dict[str, str] | None = None,
) -> Session:
    """
    Context manager that yields a ClickZetta Session and ensures it is closed.
    """

    session = create_session(
        service=service,
        instance=instance,
        workspace=workspace,
        schema=schema,
        username=username,
        password=password,
        vcluster=vcluster,
        hints=hints,
    )
    try:
        yield session
    finally:
        session.close()
