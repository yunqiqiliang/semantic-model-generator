from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from dataclasses import dataclass
from enum import Enum
from io import StringIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Final

import pandas as pd
import streamlit as st
from PIL import Image

ProgrammingError = Exception

from semantic_model_generator.data_processing.proto_utils import (
    proto_to_yaml,
    yaml_to_semantic_model,
)
from semantic_model_generator.generate_model import (
    generate_model_str_from_clickzetta,
    raw_schema_to_semantic_context,
)
from semantic_model_generator.protos import semantic_model_pb2
from semantic_model_generator.protos.semantic_model_pb2 import Dimension, Table
from semantic_model_generator.clickzetta_utils.env_vars import (  # noqa: E402
    CLICKZETTA_INSTANCE,
    CLICKZETTA_SERVICE,
    CLICKZETTA_USERNAME,
    CLICKZETTA_WORKSPACE,
    CLICKZETTA_SCHEMA,
    ACTIVE_CONFIG_PATH,
    assert_required_env_vars,
)
from semantic_model_generator.clickzetta_utils.clickzetta_connector import (
    ClickzettaConnectionProxy,
    ClickzettaConnector,
    fetch_databases,
    fetch_schemas_in_database,
    fetch_stages_in_schema,
    fetch_table_schema,
    fetch_tables_views_in_schema,
    fetch_warehouses,
    fetch_yaml_names_in_stage,
)
from semantic_model_generator.llm import get_dashscope_settings

ClickzettaConnection = ClickzettaConnectionProxy

CLICKZETTA_INSTANCE_ID = CLICKZETTA_INSTANCE or ""

# Placeholder logos (update with official ClickZetta branding when available)
LOGO_URL_LARGE: Final[str] = ""
LOGO_URL_SMALL: Final[str] = ""
_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_LOGO_PATH: Final[Path] = _REPO_ROOT / "images" / "logo.svg"


@st.cache_data(show_spinner=False)
def get_sidebar_logo_base64() -> str:
    """
    Returns a base64-encoded logo so sidebar headers can render without remote fetches.
    """
    try:
        return base64.b64encode(_LOGO_PATH.read_bytes()).decode("utf-8")
    except FileNotFoundError:
        return ""


def render_sidebar_title(
    sidebar: st.delta_generator.DeltaGenerator,
    *,
    title: str = "ClickZetta Semantic Model Generator",
    logo_width: int = 48,
    margin_bottom: int = 24,
) -> None:
    """
    Renders a consistent sidebar title block with the ClickZetta logo.
    """
    logo_b64 = get_sidebar_logo_base64()
    if not logo_b64:
        sidebar.header(title)
        sidebar.markdown(
            f"<div style='margin-bottom:{margin_bottom}px;'></div>",
            unsafe_allow_html=True,
        )
        return
    sidebar.markdown(
        (
            "<div style='display:flex; align-items:center; gap:12px;"
            f" margin-bottom:{margin_bottom}px;'>"
            f"<img src='data:image/svg+xml;base64,{logo_b64}' width='{logo_width}' style='display:block;'/>"
            f"<span style='font-weight:600;font-size:18px;'>{title}</span>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_connector() -> ClickzettaConnector:
    """
    Instantiates a ClickZetta connector using the provided credentials.
    Returns: ClickzettaConnector object
    """
    return ClickzettaConnector(max_workers=1)


def set_streamlit_location() -> bool:
    """
    Sets sis in session_state to True if the streamlit app is in SiS.
    """
    HOME = os.getenv("HOME", None)
    if HOME == "/home/udf":
        sis = True
    else:
        sis = False
    return sis


@st.experimental_dialog(title="Setup")
def env_setup_popup(missing_env_vars: list[str]) -> None:
    """
    Renders a dialog box to prompt the user to set the required connection setup.
    Args:
        missing_env_vars: A list of missing environment variables.
    """
    formatted_missing_env_vars = "\n".join(f"- **{s}**" for s in missing_env_vars)
    st.markdown(
        f"""Oops! It looks like the following required environment variables are missing: \n{formatted_missing_env_vars}\n\n
Please review the repository README for ClickZetta setup instructions and update your connection configuration accordingly. Restart this app after you've set the required environment variables."""
    )
    st.stop()


@st.cache_resource(show_spinner=False)
def get_clickzetta_connection() -> ClickzettaConnectionProxy:
    """
    Opens a ClickZetta session proxy. Cached to reuse across the app.
    """

    missing_env_vars = assert_required_env_vars()
    if missing_env_vars:
        env_setup_popup(missing_env_vars)

    return get_connector().open_connection()


@st.cache_resource(show_spinner=False)
def set_clickzetta_session(_conn: Optional[ClickzettaConnectionProxy] = None) -> None:
    """
    Stores the active ClickZetta session in Streamlit state for reuse.
    """

    conn = _conn or get_clickzetta_connection()
    st.session_state["session"] = conn.session


@st.cache_resource(show_spinner=False)
def get_available_tables(schema: str) -> list[str]:
    """
    Simple wrapper around fetch_table_names to cache the results.

    Returns: list of fully qualified table names
    """

    return fetch_tables_views_in_schema(get_clickzetta_connection().session, schema)


@st.cache_resource(show_spinner=False)
def get_available_schemas(db: str) -> list[str]:
    """
    Simple wrapper around fetch_schemas to cache the results.

    Returns: list of schema names
    """

    return fetch_schemas_in_database(get_clickzetta_connection().session, db)


@st.cache_resource(show_spinner=False)
def get_available_databases() -> list[str]:
    """
    Simple wrapper around fetch_databases to cache the results.

    Returns: list of database names
    """

    return fetch_databases(get_clickzetta_connection().session)


@st.cache_resource(show_spinner=False)
def get_available_warehouses() -> list[str]:
    """
    Simple wrapper around fetch_warehouses to cache the results.

    Returns: list of warehouse names
    """

    return fetch_warehouses(get_clickzetta_connection().session)


@st.cache_resource(show_spinner=False)
def get_available_stages(schema: str) -> List[str]:
    """
    Fetches the available volumes (legacy stage support included) from the ClickZetta workspace.

    Returns:
        List[str]: A list of available volume identifiers.
    """
    return fetch_stages_in_schema(get_clickzetta_connection(), schema)


@st.cache_resource(show_spinner=False)
def validate_table_schema(table: str, schema: Dict[str, str]) -> bool:
    table_schema = fetch_table_schema(get_clickzetta_connection().session, table)
    # validate columns names
    if set(schema.keys()) != set(table_schema.keys()):
        return False
    # validate column types
    for col_name, col_type in table_schema.items():
        if not (schema[col_name] in col_type):
            return False
    return True


@st.cache_resource(show_spinner=False)
def validate_table_exist(schema: str, table_name: str) -> bool:
    """
    Validate table exists in the ClickZetta workspace.

    Returns:
        List[str]: A list of available volumes.
    """
    table_names = fetch_tables_views_in_schema(get_clickzetta_connection().session, schema)
    table_names = [table.split(".")[2] for table in table_names]
    if table_name.upper() in table_names:
        return True
    return False


def schema_selector_container(
    db_selector: Dict[str, str], schema_selector: Dict[str, str]
) -> List[str]:
    """
    Common component that encapsulates db/schema/table selection for the admin app.
    When a db/schema/table is selected, it is saved to the session state for reading elsewhere.
    Returns: None
    """
    available_schemas = []
    available_tables = []

    # First, retrieve all databases that the user has access to.
    eval_database = st.selectbox(
        db_selector["label"],
        options=get_available_databases(),
        index=None,
        key=db_selector["key"],
    )
    if eval_database:
        # When a valid database is selected, fetch the available schemas in that database.
        try:
            available_schemas = get_available_schemas(eval_database)
        except (ValueError, ProgrammingError):
            st.error("Insufficient permissions to read from the selected database.")
            st.stop()

    eval_schema = st.selectbox(
        schema_selector["label"],
        options=available_schemas,
        index=None,
        key=schema_selector["key"],
        format_func=lambda x: format_workspace_context(x, -1),
    )
    if eval_schema:
        # When a valid schema is selected, fetch the available tables in that schema.
        try:
            available_tables = get_available_tables(eval_schema)
        except (ValueError, ProgrammingError):
            st.error("Insufficient permissions to read from the selected schema.")
            st.stop()

    return available_tables


def table_selector_container(
    db_selector: Dict[str, str],
    schema_selector: Dict[str, str],
    table_selector: Dict[str, str],
) -> Optional[str]:
    """
    Common component that encapsulates db/schema/table selection for the admin app.
    When a db/schema/table is selected, it is saved to the session state for reading elsewhere.
    Returns: None
    """
    available_schemas = []
    available_tables = []

    # First, retrieve all databases that the user has access to.
    eval_database = st.selectbox(
        db_selector["label"],
        options=get_available_databases(),
        index=None,
        key=db_selector["key"],
    )
    if eval_database:
        # When a valid database is selected, fetch the available schemas in that database.
        try:
            available_schemas = get_available_schemas(eval_database)
        except (ValueError, ProgrammingError):
            st.error("Insufficient permissions to read from the selected database.")
            st.stop()

    eval_schema = st.selectbox(
        schema_selector["label"],
        options=available_schemas,
        index=None,
        key=schema_selector["key"],
        format_func=lambda x: format_workspace_context(x, -1),
    )
    if eval_schema:
        # When a valid schema is selected, fetch the available tables in that schema.
        try:
            available_tables = get_available_tables(eval_schema)
        except (ValueError, ProgrammingError):
            st.error("Insufficient permissions to read from the selected schema.")
            st.stop()

    tables = st.selectbox(
        table_selector["label"],
        options=available_tables,
        index=None,
        key=table_selector["key"],
        format_func=lambda x: format_workspace_context(x, -1),
    )

    return tables


def stage_selector_container() -> None:
    """
    Common component that encapsulates db/schema/volume selection for the admin app.
    When a target is selected, values are saved to the session state for reuse elsewhere.
    """

    previous_mode = st.session_state.get("selected_iteration_storage_mode", "user")
    storage_mode = st.radio(
        "Storage target",
        ("user", "named"),
        index=0 if previous_mode != "named" else 1,
        format_func=lambda option: "User volume (recommended)"
        if option == "user"
        else "Named volume",
        key="selected_iteration_storage_mode",
    )

    if storage_mode == "user":
        default_dir = st.text_input(
            "User volume directory",
            value=st.session_state.get("selected_iteration_user_volume_dir", "semantic_models"),
            help="Directory path inside your user volume (omit leading and trailing slashes).",
        ).strip()

        normalized_dir = default_dir.strip("/ ")
        if not normalized_dir:
            normalized_dir = "semantic_model"
        volume_uri = "volume:user://~/"
        if normalized_dir:
            volume_uri = f"volume:user://~/{normalized_dir}/"

        st.session_state["selected_iteration_user_volume_dir"] = normalized_dir
        st.session_state["selected_iteration_database"] = "USER VOLUME"
        st.session_state["selected_iteration_schema"] = normalized_dir or "~"
        st.session_state["selected_iteration_stage"] = volume_uri
        return

    # Named volume flow (legacy stage selection retained for compatibility)
    available_schemas: List[str] = []
    available_stages: List[str] = []

    stage_database = st.selectbox(
        "Volume database",
        options=get_available_databases(),
        index=None,
        key="selected_iteration_database",
    )
    if stage_database:
        try:
            available_schemas = get_available_schemas(stage_database)
        except (ValueError, ProgrammingError):
            st.error("Insufficient permissions to read from the selected database.")
            st.stop()

    stage_schema = st.selectbox(
        "Volume schema",
        options=available_schemas,
        index=None,
        key="selected_iteration_schema",
        format_func=lambda x: format_workspace_context(x, -1),
    )
    if stage_schema:
        try:
            available_stages = get_available_stages(stage_schema)
        except (ValueError, ProgrammingError):
            st.error("Insufficient permissions to read from the selected schema.")
            st.stop()

    st.selectbox(
        "Volume",
        options=available_stages,
        index=None,
        key="selected_iteration_stage",
        format_func=lambda x: format_workspace_context(x, -1),
    )


def get_yamls_from_stage(stage: str, include_yml: bool = False) -> List[str]:
    """
    Fetches the YAML files from the specified volume.

    Args:
        stage (str): The volume identifier used to locate YAML files.
        include_yml: If True, will look for .yaml and .yml. If False, just .yaml. Defaults to False.

    Returns:
        List[str]: A list of YAML files in the specified storage target.
    """
    return fetch_yaml_names_in_stage(get_clickzetta_connection(), stage, include_yml)


def set_account_name(
    conn: ClickzettaConnectionProxy, instance_name: Optional[str] = None
) -> None:
    """
    Sets account_name in st.session_state using ClickZetta instance information.
    """

    resolved = instance_name or conn.config.get("instance") or CLICKZETTA_INSTANCE_ID
    st.session_state["account_name"] = resolved


def set_host_name(
    conn: ClickzettaConnectionProxy, service_host: Optional[str] = None
) -> None:
    """
    Sets host_name in st.session_state.
    Used to consolidate from various connection methods.
    Value only necessary for open-source implementation.
    """
    host_value = service_host or conn.host or CLICKZETTA_SERVICE or ""
    st.session_state["host_name"] = host_value


def set_user_name(
    conn: ClickzettaConnectionProxy, clickzetta_user: Optional[str] = None
) -> None:
    """
    Sets user_name in st.session_state.
    Used to consolidate from various connection methods.
    """
    if st.session_state.get("sis"):
        st.session_state["user_name"] = st.experimental_user.user_name
        return

    resolved_user = clickzetta_user or conn.config.get("username") or CLICKZETTA_USERNAME or ""
    st.session_state["user_name"] = resolved_user


class GeneratorAppScreen(str, Enum):
    """
    Enum defining different pages in the app.
    There are two full page experiences - "onboarding" and "iteration", and the builder flow is simply a popup
    that leads into the iteration flow.
    """

    ONBOARDING = "onboarding"
    ITERATION = "iteration"


def return_home_button(container=st) -> None:
    if container.button("Return to Home"):
        st.session_state["page"] = GeneratorAppScreen.ONBOARDING
        # Reset environment variables related to the semantic model, so that builder/iteration flows can start fresh.
        if "semantic_model" in st.session_state:
            del st.session_state["semantic_model"]
        if "yaml" in st.session_state:
            del st.session_state["yaml"]
        if "storage_target" in st.session_state:
            del st.session_state["storage_target"]
        st.rerun()


def update_last_validated_model() -> None:
    """Whenever user validated, update the last_validated_model to track semantic_model,
    except for verified_queries field."""
    st.session_state.last_validated_model.CopyFrom(st.session_state.semantic_model)
    # Do not save verfieid_queries field for the latest validated.
    del st.session_state.last_validated_model.verified_queries[:]


def changed_from_last_validated_model() -> bool:
    """Compare the last validated model against latest semantic model,
    except for verified_queries field."""

    for field in st.session_state.semantic_model.DESCRIPTOR.fields:
        if field.name != "verified_queries":
            model_value = getattr(st.session_state.semantic_model, field.name)
            last_validated_value = getattr(
                st.session_state.last_validated_model, field.name
            )
            if model_value != last_validated_value:
                return True
    return False


def init_session_states() -> None:
    # semantic_model stores the proto of generated semantic model using app.
    if "semantic_model" not in st.session_state:
        st.session_state.semantic_model = semantic_model_pb2.SemanticModel()
    # validated stores the status if the generated yaml has ever been validated.
    if "validated" not in st.session_state:
        st.session_state.validated = None
    # last_validated_model stores the proto (without verfied queries) from last successful validation.
    if "last_validated_model" not in st.session_state:
        st.session_state.last_validated_model = semantic_model_pb2.SemanticModel()

    # Chat display settings.
    if "chat_debug" not in st.session_state:
        st.session_state.chat_debug = False
    if "multiturn" not in st.session_state:
        st.session_state.multiturn = False

    # initialize session states for the chat page.
    if "messages" not in st.session_state:
        # messages store all chat histories
        st.session_state.messages = []
        # suggestions store suggested questions (if reject to answer) generated by the api during chat.
        st.session_state.suggestions = []
        # active_suggestion stores the active suggestion selected by the user
        st.session_state.active_suggestion = None
        # indicates if the user is editing the generated SQL for the verified query.
        st.session_state.editing = False
        # indicates if the user has confirmed his/her edits for the verified query.
        st.session_state.confirmed_edits = False


@st.experimental_dialog("Edit Dimension")  # type: ignore[misc]
def edit_dimension(table_name: str, dim: semantic_model_pb2.Dimension) -> None:
    """
    Renders a dialog box to edit an existing dimension.
    """
    key_prefix = f"{table_name}-{dim.name}"
    dim.name = st.text_input("Name", dim.name, key=f"{key_prefix}-edit-dim-name")
    dim.expr = st.text_input(
        "SQL Expression", dim.expr, key=f"{key_prefix}-edit-dim-expr"
    )
    dim.description = st.text_area(
        "Description", dim.description, key=f"{key_prefix}-edit-dim-description"
    )
    # Allow users to edit synonyms through a data_editor.
    synonyms_df = st.data_editor(
        pd.DataFrame(list(dim.synonyms), columns=["Synonyms"]),
        num_rows="dynamic",
        key=f"{key_prefix}-edit-dim-synonyms",
    )
    # Store the current values in data_editor in the protobuf.
    del dim.synonyms[:]
    for _, row in synonyms_df.iterrows():
        if row["Synonyms"]:
            dim.synonyms.append(row["Synonyms"])

    # TODO(nsehrawat): Change to a select box with a list of all data types.
    dim.data_type = st.text_input(
        "Data type", dim.data_type, key=f"{key_prefix}-edit-dim-datatype"
    )
    dim.unique = st.checkbox(
        "Does it have unique values?",
        value=dim.unique,
        key=f"{key_prefix}-edit-dim-unique",
    )
    # Allow users to edit sample values through a data_editor.
    sample_values_df = st.data_editor(
        pd.DataFrame(list(dim.sample_values), columns=["Sample Values"]),
        num_rows="dynamic",
        key=f"{key_prefix}-edit-dim-sample-values",
    )
    # Store the current values in data_editor in the protobuf.
    del dim.sample_values[:]
    for _, row in sample_values_df.iterrows():
        if row["Sample Values"]:
            dim.sample_values.append(row["Sample Values"])

    if st.button("Save"):
        st.rerun()


@st.experimental_dialog("Add Dimension")  # type: ignore[misc]
def add_dimension(table: semantic_model_pb2.Table) -> None:
    """
    Renders a dialog box to add a new dimension.
    """
    dim = Dimension()
    dim.name = st.text_input("Name", key=f"{table.name}-add-dim-name")
    dim.expr = st.text_input("SQL Expression", key=f"{table.name}-add-dim-expr")
    dim.description = st.text_area(
        "Description", key=f"{table.name}-add-dim-description"
    )
    synonyms_df = st.data_editor(
        pd.DataFrame(list(dim.synonyms), columns=["Synonyms"]),
        num_rows="dynamic",
        key=f"{table.name}-add-dim-synonyms",
    )
    for _, row in synonyms_df.iterrows():
        if row["Synonyms"]:
            dim.synonyms.append(row["Synonyms"])

    dim.data_type = st.text_input("Data type", key=f"{table.name}-add-dim-datatype")
    dim.unique = st.checkbox(
        "Does it have unique values?", key=f"{table.name}-add-dim-unique"
    )
    sample_values_df = st.data_editor(
        pd.DataFrame(list(dim.sample_values), columns=["Sample Values"]),
        num_rows="dynamic",
        key=f"{table.name}-add-dim-sample-values",
    )
    del dim.sample_values[:]
    for _, row in sample_values_df.iterrows():
        if row["Sample Values"]:
            dim.sample_values.append(row["Sample Values"])

    if st.button("Add"):
        table.dimensions.append(dim)
        st.rerun()


@st.experimental_dialog("Edit Measure/Fact")  # type: ignore[misc]
def edit_measure(table_name: str, measure: semantic_model_pb2.Fact) -> None:
    """
    Renders a dialog box to edit an existing measure.
    """
    key_prefix = f"{table_name}-{measure.name}"
    measure.name = st.text_input(
        "Name", measure.name, key=f"{key_prefix}-edit-measure-name"
    )
    measure.expr = st.text_input(
        "SQL Expression", measure.expr, key=f"{key_prefix}-edit-measure-expr"
    )
    measure.description = st.text_area(
        "Description", measure.description, key=f"{key_prefix}-edit-measure-description"
    )
    synonyms_df = st.data_editor(
        pd.DataFrame(list(measure.synonyms), columns=["Synonyms"]),
        num_rows="dynamic",
        key=f"{key_prefix}-edit-measure-synonyms",
    )
    del measure.synonyms[:]
    for _, row in synonyms_df.iterrows():
        if row["Synonyms"]:
            measure.synonyms.append(row["Synonyms"])

    measure.data_type = st.text_input(
        "Data type", measure.data_type, key=f"{key_prefix}-edit-measure-data-type"
    )

    aggr_options = semantic_model_pb2.AggregationType.keys()
    # Replace the 'aggregation_type_unknown' string with an empty string for a better display of options.
    aggr_options[0] = ""
    default_aggregation_idx = next(
        (
            i
            for i, s in enumerate(semantic_model_pb2.AggregationType.values())
            if s == measure.default_aggregation
        ),
        0,
    )

    default_aggregation = st.selectbox(
        "Default Aggregation",
        aggr_options,
        index=default_aggregation_idx,
        key=f"{key_prefix}-edit-measure-default-aggregation",
    )
    if default_aggregation:
        try:
            measure.default_aggregation = semantic_model_pb2.AggregationType.Value(
                default_aggregation
            )  # type: ignore[assignment]
        except ValueError as e:
            st.error(f"Invalid default_aggregation: {e}")
    else:
        measure.default_aggregation = (
            semantic_model_pb2.AggregationType.aggregation_type_unknown
        )

    sample_values_df = st.data_editor(
        pd.DataFrame(list(measure.sample_values), columns=["Sample Values"]),
        num_rows="dynamic",
        key=f"{key_prefix}-edit-measure-sample-values",
    )
    del measure.sample_values[:]
    for _, row in sample_values_df.iterrows():
        if row["Sample Values"]:
            measure.sample_values.append(row["Sample Values"])

    if st.button("Save"):
        st.rerun()


@st.experimental_dialog("Add Measure/Fact")  # type: ignore[misc]
def add_measure(table: semantic_model_pb2.Table) -> None:
    """
    Renders a dialog box to add a new measure.
    """
    with st.form(key="add-measure"):
        measure = semantic_model_pb2.Fact()
        measure.name = st.text_input("Name", key=f"{table.name}-add-measure-name")
        measure.expr = st.text_input(
            "SQL Expression", key=f"{table.name}-add-measure-expr"
        )
        measure.description = st.text_area(
            "Description", key=f"{table.name}-add-measure-description"
        )
        synonyms_df = st.data_editor(
            pd.DataFrame(list(measure.synonyms), columns=["Synonyms"]),
            num_rows="dynamic",
            key=f"{table.name}-add-measure-synonyms",
        )
        del measure.synonyms[:]
        for _, row in synonyms_df.iterrows():
            if row["Synonyms"]:
                measure.synonyms.append(row["Synonyms"])

        measure.data_type = st.text_input(
            "Data type", key=f"{table.name}-add-measure-data-type"
        )
        aggr_options = semantic_model_pb2.AggregationType.keys()
        # Replace the 'aggregation_type_unknown' string with an empty string for a better display of options.
        aggr_options[0] = ""
        default_aggregation = st.selectbox(
            "Default Aggregation",
            aggr_options,
            key=f"{table.name}-edit-measure-default-aggregation",
        )
        if default_aggregation:
            try:
                measure.default_aggregation = semantic_model_pb2.AggregationType.Value(
                    default_aggregation
                )  # type: ignore[assignment]
            except ValueError as e:
                st.error(f"Invalid default_aggregation: {e}")

        sample_values_df = st.data_editor(
            pd.DataFrame(list(measure.sample_values), columns=["Sample Values"]),
            num_rows="dynamic",
            key=f"{table.name}-add-measure-sample-values",
        )
        del measure.sample_values[:]
        for _, row in sample_values_df.iterrows():
            if row["Sample Values"]:
                measure.sample_values.append(row["Sample Values"])

        add_button = st.form_submit_button("Add")

    if add_button:
        table.measures.append(measure)
        st.rerun()


@st.experimental_dialog("Edit Time Dimension")  # type: ignore[misc]
def edit_time_dimension(
    table_name: str, tdim: semantic_model_pb2.TimeDimension
) -> None:
    """
    Renders a dialog box to edit a time dimension.
    """
    key_prefix = f"{table_name}-{tdim.name}"
    tdim.name = st.text_input("Name", tdim.name, key=f"{key_prefix}-edit-tdim-name")
    tdim.expr = st.text_input(
        "SQL Expression", tdim.expr, key=f"{key_prefix}-edit-tdim-expr"
    )
    tdim.description = st.text_area(
        "Description",
        tdim.description,
        key=f"{key_prefix}-edit-tdim-description",
    )
    synonyms_df = st.data_editor(
        pd.DataFrame(list(tdim.synonyms), columns=["Synonyms"]),
        num_rows="dynamic",
        key=f"{key_prefix}-tdim-edit-measure-synonyms",
    )
    del tdim.synonyms[:]
    for _, row in synonyms_df.iterrows():
        if row["Synonyms"]:
            tdim.synonyms.append(row["Synonyms"])

    tdim.data_type = st.text_input(
        "Data type", tdim.data_type, key=f"{key_prefix}-edit-tdim-datatype"
    )
    tdim.unique = st.checkbox("Does it have unique values?", value=tdim.unique)
    sample_values_df = st.data_editor(
        pd.DataFrame(list(tdim.sample_values), columns=["Sample Values"]),
        num_rows="dynamic",
        key=f"{key_prefix}-edit-tdim-sample-values",
    )
    del tdim.sample_values[:]
    for _, row in sample_values_df.iterrows():
        if row["Sample Values"]:
            tdim.sample_values.append(row["Sample Values"])

    if st.button("Save"):
        st.rerun()


@st.experimental_dialog("Add Time Dimension")  # type: ignore[misc]
def add_time_dimension(table: semantic_model_pb2.Table) -> None:
    """
    Renders a dialog box to add a new time dimension.
    """
    tdim = semantic_model_pb2.TimeDimension()
    tdim.name = st.text_input("Name", key=f"{table.name}-add-tdim-name")
    tdim.expr = st.text_input("SQL Expression", key=f"{table.name}-add-tdim-expr")
    tdim.description = st.text_area(
        "Description", key=f"{table.name}-add-tdim-description"
    )
    synonyms_df = st.data_editor(
        pd.DataFrame(list(tdim.synonyms), columns=["Synonyms"]),
        num_rows="dynamic",
        key=f"{table.name}-add-tdim-synonyms",
    )
    del tdim.synonyms[:]
    for _, row in synonyms_df.iterrows():
        if row["Synonyms"]:
            tdim.synonyms.append(row["Synonyms"])

    # TODO(nsehrawat): Change the set of allowed data types here.
    tdim.data_type = st.text_input("Data type", key=f"{table.name}-add-tdim-data-types")
    tdim.unique = st.checkbox(
        "Does it have unique values?", key=f"{table.name}-add-tdim-unique"
    )
    sample_values_df = st.data_editor(
        pd.DataFrame(list(tdim.sample_values), columns=["Sample Values"]),
        num_rows="dynamic",
        key=f"{table.name}-add-tdim-sample-values",
    )
    del tdim.sample_values[:]
    for _, row in sample_values_df.iterrows():
        if row["Sample Values"]:
            tdim.sample_values.append(row["Sample Values"])

    if st.button("Add", key=f"{table.name}-add-tdim-add"):
        table.time_dimensions.append(tdim)
        st.rerun()


def delete_dimension(table: semantic_model_pb2.Table, idx: int) -> None:
    """
    Inline deletes the dimension at a particular index in a Table protobuf.
    """
    if len(table.dimensions) < idx:
        return
    del table.dimensions[idx]


def delete_measure(table: semantic_model_pb2.Table, idx: int) -> None:
    """
    Inline deletes the measure at a particular index in a Table protobuf.
    """
    if len(table.measures) < idx:
        return
    del table.measures[idx]


def delete_time_dimension(table: semantic_model_pb2.Table, idx: int) -> None:
    """
    Inline deletes the time dimension at a particular index in a Table protobuf.
    """
    if len(table.time_dimensions) < idx:
        return
    del table.time_dimensions[idx]


def display_table(table_name: str) -> None:
    """
    Display all the data related to a logical table.
    """
    for t in st.session_state.semantic_model.tables:
        if t.name == table_name:
            table: semantic_model_pb2.Table = t
            break

    st.write("#### Table metadata")
    table.name = st.text_input("Table Name", table.name)
    fqn_columns = st.columns(3)
    with fqn_columns[0]:
        table.base_table.database = st.text_input(
            "Physical Database",
            table.base_table.database,
            key=f"{table_name}-base_database",
        )
    with fqn_columns[1]:
        table.base_table.schema = st.text_input(
            "Physical Schema",
            table.base_table.schema,
            key=f"{table_name}-base_schema",
        )
    with fqn_columns[2]:
        table.base_table.table = st.text_input(
            "Physical Table", table.base_table.table, key=f"{table_name}-base_table"
        )

    table.description = st.text_area(
        "Description", table.description, key=f"{table_name}-description"
    )

    synonyms_df = st.data_editor(
        pd.DataFrame(list(table.synonyms), columns=["Synonyms"]),
        num_rows="dynamic",
        key=f"{table_name}-synonyms",
        use_container_width=True,
    )
    del table.synonyms[:]
    for idx, row in synonyms_df.iterrows():
        if row["Synonyms"]:
            table.synonyms.append(row["Synonyms"])

    st.write("#### Dimensions")
    header = ["Name", "Expression", "Data Type"]
    header_cols = st.columns(len(header) + 1)
    for i, h in enumerate(header):
        header_cols[i].write(f"###### {h}")

    for idx, dim in enumerate(table.dimensions):
        cols = st.columns(len(header) + 1)
        cols[0].write(dim.name)
        cols[1].write(dim.expr)
        cols[2].write(dim.data_type)
        with cols[-1]:
            subcols = st.columns(2)
            if subcols[0].button(
                "Edit",
                key=f"{table_name}-edit-dimension-{idx}",
            ):
                edit_dimension(table_name, dim)
            subcols[1].button(
                "Delete",
                key=f"{table_name}-delete-dimension-{idx}",
                on_click=delete_dimension,
                args=(
                    table,
                    idx,
                ),
            )

    if st.button("Add Dimension", key=f"{table_name}-add-dimension"):
        add_dimension(table)

    st.write("#### Measures")
    header_cols = st.columns(len(header) + 1)
    for i, h in enumerate(header):
        header_cols[i].write(f"###### {h}")

    for idx, measure in enumerate(table.measures):
        cols = st.columns(len(header) + 1)
        cols[0].write(measure.name)
        cols[1].write(measure.expr)
        cols[2].write(measure.data_type)
        with cols[-1]:
            subcols = st.columns(2)
            if subcols[0].button("Edit", key=f"{table_name}-edit-measure-{idx}"):
                edit_measure(table_name, measure)
            subcols[1].button(
                "Delete",
                key=f"{table_name}-delete-measure-{idx}",
                on_click=delete_measure,
                args=(
                    table,
                    idx,
                ),
            )

    if st.button("Add Measure", key=f"{table_name}-add-measure"):
        add_measure(table)

    st.write("#### Time Dimensions")
    header_cols = st.columns(len(header) + 1)
    for i, h in enumerate(header):
        header_cols[i].write(f"###### {h}")

    for idx, tdim in enumerate(table.time_dimensions):
        cols = st.columns(len(header) + 1)
        cols[0].write(tdim.name)
        cols[1].write(tdim.expr)
        cols[2].write(tdim.data_type)
        with cols[-1]:
            subcols = st.columns(2)
            if subcols[0].button("Edit", key=f"{table_name}-edit-tdim-{idx}"):
                edit_time_dimension(table_name, tdim)
            subcols[1].button(
                "Delete",
                key=f"{table_name}-delete-tdim-{idx}",
                on_click=delete_time_dimension,
                args=(
                    table,
                    idx,
                ),
            )

    if st.button("Add Time Dimension", key=f"{table_name}-add-tdim"):
        add_time_dimension(table)


@st.experimental_dialog("Add Table")  # type: ignore[misc]
def add_new_table() -> None:
    """
    Renders a dialog box to add a new logical table.
    """
    table = Table()
    table.name = st.text_input("Table Name")
    for t in st.session_state.semantic_model.tables:
        if t.name == table.name:
            st.error(f"Table called '{table.name}' already exists")

    table.base_table.database = st.text_input("Physical Database")
    table.base_table.schema = st.text_input("Physical Schema")
    table.base_table.table = st.text_input("Physical Table")
    st.caption(":gray[Synonyms (hover the table to add new rows!)]")
    synonyms_df = st.data_editor(
        pd.DataFrame(columns=["Synonyms"]),
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
    )
    for _, row in synonyms_df.iterrows():
        if row["Synonyms"]:
            table.synonyms.append(row["Synonyms"])
    table.description = st.text_area("Description", key="add-new-table-description")
    if st.button("Add"):
        with st.spinner(text="Fetching table details from database ..."):
            try:
                semantic_model = raw_schema_to_semantic_context(
                    base_tables=[
                        f"{table.base_table.database}.{table.base_table.schema}.{table.base_table.table}"
                    ],
                    semantic_model_name="foo",  # A placeholder name that's not used anywhere.
                    conn=get_clickzetta_connection(),
                )
            except Exception as ex:
                st.error(f"Error adding table: {ex}")
                return
            table.dimensions.extend(semantic_model.tables[0].dimensions)
            table.measures.extend(semantic_model.tables[0].measures)
            table.time_dimensions.extend(semantic_model.tables[0].time_dimensions)
            for t in st.session_state.semantic_model.tables:
                if t.name == table.name:
                    st.error(f"Table called '{table.name}' already exists")
                    return
        st.session_state.semantic_model.tables.append(table)
        st.rerun()


def display_semantic_model() -> None:
    """
    Renders the entire semantic model.
    """
    semantic_model = st.session_state.semantic_model
    with st.form(border=False, key="create"):
        name = st.text_input(
            "Name",
            semantic_model.name,
            placeholder="My semantic model",
        )

        description = st.text_area(
            "Description",
            semantic_model.description,
            key="display-semantic-model-description",
            placeholder="The model describes the data and metrics available for Foocorp",
        )

        left, right = st.columns((1, 4))
        if left.form_submit_button("Create", use_container_width=True):
            st.session_state.semantic_model.name = name
            st.session_state.semantic_model.description = description
            st.session_state["next_is_unlocked"] = True
            right.success("Successfully created model. Updating...")
            time.sleep(1.5)
            st.rerun()


def edit_semantic_model() -> None:
    st.write("#### Tables")
    for t in st.session_state.semantic_model.tables:
        with st.expander(t.name):
            display_table(t.name)
    if st.button("Add Table"):
        add_new_table()


def import_yaml() -> None:
    """
    Renders a page to import an existing yaml file.
    """
    uploaded_file = st.file_uploader(
        "Choose a semantic model YAML file",
        type=[".yaml", ".yml"],
        accept_multiple_files=False,
    )
    pb: Optional[semantic_model_pb2.SemanticModel] = None

    if uploaded_file is not None:
        try:
            yaml_str = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            pb = yaml_to_semantic_model(yaml_str)
        except Exception as ex:
            st.error(f"Failed to import: {ex}")
            return
        if pb is None:
            st.error("Failed to import, did you choose a file?")
            return

        st.session_state["semantic_model"] = pb
        st.success(f"Successfully imported **{pb.name}**!")
        st.session_state["next_is_unlocked"] = True
        if "yaml_just_imported" not in st.session_state:
            st.session_state["yaml_just_imported"] = True
            st.rerun()


@st.experimental_dialog("Model YAML", width="large")  # type: ignore
def show_yaml_in_dialog() -> None:
    yaml = proto_to_yaml(st.session_state.semantic_model)
    st.code(
        yaml,
        language="yaml",
        line_numbers=True,
    )


def upload_yaml(file_name: str) -> None:
    """util to upload the semantic model."""
    yaml = proto_to_yaml(st.session_state.semantic_model)
    conn = get_clickzetta_connection()

    with tempfile.TemporaryDirectory() as temp_dir:
        tmp_file_path = os.path.join(temp_dir, f"{file_name}.yaml")

        with open(tmp_file_path, "w", encoding="utf-8") as temp_file:
            temp_file.write(yaml)

        destination_uri = st.session_state.storage_target.target_uri(
            f"{file_name}.yaml"
        )
        conn.session.file.put(
            tmp_file_path,
            destination_uri,
            auto_compress=False,
            overwrite=True,
        )


def delete_yaml(file_name: str, stage: StorageTarget) -> None:
    """Remove a semantic model YAML from the specified storage target."""

    conn = get_clickzetta_connection()
    target_uri = stage.target_uri(file_name)
    conn.session.file.delete(target_uri)


def validate_and_upload_tmp_yaml(conn: ClickzettaConnectionProxy) -> None:
    """
    Validate the semantic model.
    If successfully validated, upload a temp file into the working volume, to allow chatting and adding VQR against it.
    """
    from semantic_model_generator.validate_model import validate

    yaml_str = proto_to_yaml(st.session_state.semantic_model)
    try:
        # whenever valid, upload to temp volume path.
        validate(yaml_str, conn)
        # upload_yaml(_TMP_FILE_NAME)
        st.session_state.validated = True
        update_last_validated_model()
    except Exception as e:
        st.warning(f"Invalid YAML: {e} please fix!")

    st.success("Successfully validated your model!")
    st.session_state["next_is_unlocked"] = True


def semantic_model_exists() -> bool:
    if "semantic_model" in st.session_state:
        if hasattr(st.session_state.semantic_model, "name"):
            if isinstance(st.session_state.semantic_model.name, str):
                model_name: str = st.session_state.semantic_model.name.strip()
                return model_name != ""
    return False


def stage_exists() -> bool:
    return "storage_target" in st.session_state


def model_is_validated() -> bool:
    if semantic_model_exists():
        return st.session_state.validated  # type: ignore
    return False


def download_yaml(file_name: str, stage: StorageTarget) -> str:
    """util to download a semantic YAML from a stage or volume."""
    conn = get_clickzetta_connection()

    with tempfile.TemporaryDirectory() as temp_dir:
        # Downloads the YAML to {temp_dir}/{file_name}.
        source_uri = stage.target_uri(file_name)
        conn.session.file.get(source_uri, temp_dir)

        tmp_file_path = os.path.join(temp_dir, f"{file_name}")
        with open(tmp_file_path, "r", encoding="utf-8") as temp_file:
            # Read the raw contents from {temp_dir}/{file_name} and return it as a string.
            yaml_str = temp_file.read()
            return yaml_str


def get_sit_query_tag(
    vendor: Optional[str] = None, action: Optional[str] = None
) -> str:
    """
    Returns SIT query tag.
    Returns: str
    """

    query_tag = {
        "origin": "sf_sit",
        "name": "skimantics",
        "version": {"major": 1, "minor": 0},
        "attributes": {"vendor": vendor, "action": action},
    }
    return json.dumps(query_tag)


def set_sit_query_tag(
    conn: ClickzettaConnectionProxy,
    vendor: Optional[str] = None,
    action: Optional[str] = None,
) -> None:
    """
    Placeholder for query tagging; ClickZetta currently does not expose this
    functionality, so this is a no-op.

    Returns: None
    """
    _ = conn
    _ = vendor
    _ = action


def set_table_comment(
    conn: ClickzettaConnectionProxy,
    tablename: str,
    comment: str,
    table_type: Optional[str] = None,
) -> None:
    """
    Sets comment on provided table.
    Returns: None
    """
    if table_type is None:
        table_type = ""
    query = f"ALTER {table_type} TABLE {tablename} SET COMMENT = '{comment}'"
    conn.cursor().execute(query)


def render_image(image_file: str, size: tuple[int, int]) -> None:
    """
    Renders image in streamlit app with custom width x height by pixel.
    """
    image = Image.open(image_file)
    new_image = image.resize(size)
    st.image(new_image)


def format_workspace_context(context: str, index: Optional[int] = None) -> str:
    """
    Extracts the desired part of a fully qualified ClickZetta identifier.
    """
    if index and "." in context:
        split_context = context.split(".")
        try:
            return split_context[index]
        except IndexError:  # Return final segment if mis-typed index
            return split_context[-1]
    else:
        return context


def check_valid_session_state_values(vars: list[str]) -> bool:
    """
    Returns False if any vars are not found or None in st.session_state.
    Args:
        vars (list[str]): List of variables to check in st.session_state

    Returns: bool
    """
    empty_vars = []
    for var in vars:
        if var not in st.session_state:
            empty_vars.append(var)
    if empty_vars:
        st.error(f"Please enter values for {vars}.")
        return False
    else:
        return True


def run_cortex_complete(
    conn: ClickzettaConnection,
    model: str,
    prompt: str,
    prompt_args: Optional[dict[str, Any]] = None,
) -> str | None:
    _ = (conn, model, prompt, prompt_args)
    st.info("Cortex completion is not available in the ClickZetta environment.")
    return None


def input_semantic_file_name() -> str:
    """
    Prompts the user to input the name of the semantic model they are creating.
    Returns:
        str: The name of the semantic model.
    """

    model_name = st.text_input(
        "Semantic Model Name (no .yaml suffix)",
        help="The name of the semantic model you are creating. This is separate from the filename, which we will set later.",
    )
    return model_name


def input_sample_value_num() -> int:
    """
    Function to prompt the user to input the maximum number of sample values per column.
    Returns:
        int: The maximum number of sample values per column.
    """

    sample_values: int = st.selectbox(  # type: ignore
        "Maximum number of sample values per column",
        list(range(1, 40)),
        index=2,
        help="Specifies the maximum number of distinct sample values we fetch for each column. We suggest keeping this number as low as possible to reduce latency.",
    )
    return sample_values


def run_generate_model_str_from_clickzetta(
    model_name: str,
    sample_values: int,
    base_tables: list[str],
    allow_joins: Optional[bool] = True,
    enrich_with_llm: bool = False,
) -> None:
    """
    Runs generate_model_str_from_clickzetta to generate the semantic shell.
    Args:
        model_name (str): Semantic file name (without .yaml suffix).
        sample_values (int): Number of sample values to provide for each table in generation.
        base_tables (list[str]): List of fully-qualified ClickZetta tables to include in the semantic model.

    Returns: None
    """

    if not model_name:
        raise ValueError("Please provide a name for your semantic model.")
    elif not base_tables:
        raise ValueError("Please select at least one table to proceed.")
    else:
        with st.spinner("Generating model. This may take minutes ..."):
            connection = get_clickzetta_connection()
            yaml_str = generate_model_str_from_clickzetta(
                base_tables=base_tables,
                semantic_model_name=model_name,
                n_sample_values=sample_values,  # type: ignore
                conn=connection.session,
                allow_joins=allow_joins,
                enrich_with_llm=enrich_with_llm,
            )

            st.session_state["yaml"] = yaml_str


@dataclass
class AppMetadata:
    """
    Metadata about the active semantic model and environment variables
    being in used in the app session.
    """

    @property
    def user(self) -> str:
        return st.session_state.get("user_name") or CLICKZETTA_USERNAME or "Unknown"

    @property
    def volume(self) -> str:
        if stage_exists():
            storage = st.session_state.storage_target
            if storage.is_volume:
                return storage.stage_name
            return f"{storage.stage_database}.{storage.stage_schema}.{storage.stage_name}"
        # Fallback to selected iteration volume if available.
        selected = st.session_state.get("selected_iteration_stage")
        if isinstance(selected, str) and selected.strip():
            return selected.strip()
        return "volume:user://~/semantic_models/"

    @property
    def model(self) -> str:
        if semantic_model_exists():
            return st.session_state.semantic_model.name  # type: ignore
        file_name = st.session_state.get("file_name")
        if isinstance(file_name, str) and file_name.strip():
            return file_name.strip()
        return "Not loaded"

    @property
    def instance(self) -> str:
        return (
            st.session_state.get("account_name")
            or CLICKZETTA_INSTANCE or ""
        )

    @property
    def service(self) -> str:
        return (
            st.session_state.get("host_name")
            or CLICKZETTA_SERVICE
            or ""
        )

    @property
    def workspace(self) -> str:
        return (
            st.session_state.get("workspace_name")
            or CLICKZETTA_WORKSPACE
            or ""
        )

    @property
    def schema(self) -> str:
        return (
            st.session_state.get("schema_name")
            or CLICKZETTA_SCHEMA
            or ""
        )

    @property
    def config_path(self) -> str:
        return ACTIVE_CONFIG_PATH or "Not found"

    def llm_model(self) -> str:
        settings = get_dashscope_settings()
        return settings.model if settings else "Not configured"

    def llm_base_url(self) -> str:
        settings = get_dashscope_settings()
        if settings and settings.base_url:
            return settings.base_url
        return "Default"

    def to_dict(self) -> dict[str, str]:
        return {
            "User": self.user,
            "Volume": self.volume,
            "Semantic Model": self.model,
        }

    def connection_dict(self) -> dict[str, str]:
        return {
            "Instance": self.instance or "Unknown",
            "Service": self.service or "Unknown",
            "Workspace": self.workspace or "Unknown",
            "Schema": self.schema or "Unknown",
        }

    def llm_dict(self) -> dict[str, str]:
        settings = get_dashscope_settings()
        effective_model = "qwen-plus-latest"
        base_url = "Default"
        if settings and settings.base_url:
            base_url = settings.base_url
        return {
            "Model": effective_model,
            "Base URL": base_url,
        }

    def show_as_dataframe(self) -> None:
        data = self.to_dict()
        st.dataframe(
            data,
            column_config={"value": st.column_config.Column(label="Value")},
            use_container_width=True,
        )


@dataclass
class StorageTarget:
    stage_database: str
    stage_schema: str
    stage_name: str

    @property
    def is_volume(self) -> bool:
        return self.stage_name.startswith("volume:")

    def target_uri(self, file_name: str) -> str:
        if self.is_volume:
            base = self.stage_name.rstrip("/")
            separator = "" if base.endswith("/") else "/"
            return f"{base}{separator}{file_name}"
        return f"@{self.stage_name}/{file_name}"

    def to_dict(self) -> dict[str, str]:
        if self.is_volume:
            return {"Volume": self.stage_name}
        return {
            "Database": self.stage_database,
            "Schema": self.stage_schema,
            "Volume": self.stage_name,
        }
