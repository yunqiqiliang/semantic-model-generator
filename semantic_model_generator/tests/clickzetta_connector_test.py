from importlib import reload
from unittest import mock

import pandas as pd

from semantic_model_generator.clickzetta_utils import env_vars
from semantic_model_generator.clickzetta_utils import clickzetta_connector as connector


def test_fetch_stages_includes_user_volume(monkeypatch):
    data = pd.DataFrame({"name": ["shared_stage"]})
    with mock.patch.object(
        connector, "_execute_query_to_pandas", return_value=data
    ):
        stages = connector.fetch_stages_in_schema(
            connection=mock.MagicMock(), schema_name="WORKSPACE.SCHEMA"
        )
    assert stages[0] == "volume:user://~/semantic_models/"
    assert "shared_stage" in stages


def test_fetch_yaml_names_in_user_volume(monkeypatch):
    data = pd.DataFrame(
        {
            "relative_path": [
                "semantic_models/example.yaml",
                "semantic_models/duplicate.yaml",
                "semantic_models/duplicate.yaml",
            ]
        }
    )
    with mock.patch.object(
        connector, "_execute_query_to_pandas", return_value=data
    ):
        files = connector.fetch_yaml_names_in_stage(
            connection=mock.MagicMock(),
            stage="volume:user://~/semantic_models/",
            include_yml=True,
        )
    assert files == ["example.yaml", "duplicate.yaml"]


def test_build_base_connection_config_includes_hints(monkeypatch):
    monkeypatch.setenv("CLICKZETTA_SERVICE", "svc")
    monkeypatch.setenv("CLICKZETTA_INSTANCE", "inst")
    monkeypatch.setenv("CLICKZETTA_WORKSPACE", "ws")
    monkeypatch.setenv("CLICKZETTA_SCHEMA", "PUBLIC")
    monkeypatch.setenv("CLICKZETTA_USERNAME", "user")
    monkeypatch.setenv("CLICKZETTA_PASSWORD", "secret")

    reload(env_vars)
    config = env_vars.build_base_connection_config()
    assert config["service"] == "svc"
    assert "hints" in config


def test_get_valid_columns_falls_back_to_show_columns():
    class DummyResult:
        def __init__(self, df: pd.DataFrame):
            self._df = df

        def to_pandas(self) -> pd.DataFrame:
            return self._df

    def sql_side_effect(query: str):
        if "SHOW COLUMNS" in query:
            df = pd.DataFrame(
                {
                    "SCHEMA_NAME": ["TPCH_100G"],
                    "TABLE_NAME": ["PARTSUPP"],
                    "COLUMN_NAME": ["PS_PARTKEY"],
                    "DATA_TYPE": ["NUMBER"],
                    "COMMENT": [""],
                }
            )
            return DummyResult(df)
        raise Exception("information_schema unavailable")

    session = mock.MagicMock()
    connector._CATALOG_CATEGORY_CACHE.clear()
    with mock.patch.object(connector, "_catalog_category", return_value="SHARED"):
        session.sql.side_effect = sql_side_effect

        df = connector.get_valid_schemas_tables_columns_df(
            session=session,
            workspace="CLICKZETTA_SAMPLE_DATA",
            table_schema="TPCH_100G",
            table_names=["PARTSUPP"],
        )
    assert not df.empty
    assert df["TABLE_NAME"].iloc[0] == "PARTSUPP"
    assert df["COLUMN_NAME"].iloc[0] == "PS_PARTKEY"
