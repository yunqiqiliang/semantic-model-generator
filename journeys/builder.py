import streamlit as st
from loguru import logger

from app_utils.shared_utils import (
    GeneratorAppScreen,
    ProgrammingError,
    format_workspace_context,
    get_available_databases,
    get_available_schemas,
    get_available_tables,
    input_sample_value_num,
    input_semantic_file_name,
    run_generate_model_str_from_clickzetta,
)
from semantic_model_generator.llm import is_llm_available


def update_schemas_and_tables() -> None:
    """
    Callback to run when the selected databases change. Ensures that if a database is deselected, the corresponding
    schemas and tables are also deselected.
    Returns: None

    """
    databases = st.session_state["selected_databases"]

    # Fetch the available schemas for the selected databases
    schemas = []
    for db in databases:
        try:
            schemas.extend(get_available_schemas(db))
        except ProgrammingError:
            logger.info(
                f"Insufficient permissions to read from database {db}, skipping"
            )

    st.session_state["available_schemas"] = schemas

    # Enforce that the previously selected schemas are still valid
    valid_selected_schemas = [
        schema for schema in st.session_state["selected_schemas"] if schema in schemas
    ]
    st.session_state["selected_schemas"] = valid_selected_schemas
    update_tables()


def update_tables() -> None:
    """
    Callback to run when the selected schemas change. Ensures that if a schema is deselected, the corresponding
    tables are also deselected.
    """
    schemas = st.session_state["selected_schemas"]

    # Fetch the available tables for the selected schemas
    tables = []
    for schema in schemas:
        try:
            tables.extend(get_available_tables(schema))
        except ProgrammingError as exc:
            logger.warning(
                "Unable to list tables for {}: {}", schema, exc
            )
    logger.debug("Collected {} objects across schemas {}", len(tables), schemas)
    st.session_state["available_tables"] = tables

    # Enforce that the previously selected tables are still valid
    valid_selected_tables = [
        table for table in st.session_state["selected_tables"] if table in tables
    ]
    st.session_state["selected_tables"] = valid_selected_tables


@st.experimental_dialog("Selecting your tables", width="large")
def table_selector_dialog() -> None:
    st.write(
        "Please fill out the following fields to start building your semantic model."
    )
    model_name = input_semantic_file_name()
    sample_values = input_sample_value_num()
    st.markdown("")

    if "selected_databases" not in st.session_state:
        st.session_state["selected_databases"] = []

    if "selected_schemas" not in st.session_state:
        st.session_state["selected_schemas"] = []

    if "selected_tables" not in st.session_state:
        st.session_state["selected_tables"] = []

    with st.spinner("Loading databases..."):
        available_databases = get_available_databases()

    st.multiselect(
        label="Databases",
        options=available_databases,
        placeholder="Select the databases that contain the tables you'd like to include in your semantic model.",
        on_change=update_schemas_and_tables,
        key="selected_databases",
        # default=st.session_state.get("selected_databases", []),
    )

    st.multiselect(
        label="Schemas",
        options=st.session_state.get("available_schemas", []),
        placeholder="Select the schemas that contain the tables you'd like to include in your semantic model.",
        on_change=update_tables,
        key="selected_schemas",
        format_func=lambda x: format_workspace_context(x, -1),
    )

    st.multiselect(
        label="Tables",
        options=st.session_state.get("available_tables", []),
        placeholder="Select the tables you'd like to include in your semantic model.",
        key="selected_tables",
        format_func=lambda x: format_workspace_context(x, -1),
    )

    st.markdown("<div style='margin: 240px;'></div>", unsafe_allow_html=True)
    experimental_features = st.checkbox(
        "Enable joins (optional)",
        help="Checking this box will enable you to add/edit join paths in your semantic model. If enabling this setting, please ensure that the required ClickZetta parameters are enabled for your workspace. Reach out to your account team for access.",
        value=True,
    )

    st.session_state["experimental_features"] = experimental_features

    llm_available = is_llm_available()
    llm_help = (
        "When enabled, DashScope will suggest business-friendly aliases, expand table and column descriptions, "
        "and generate starter questions to accelerate review."
    )
    if not llm_available:
        llm_help = (
            f"{llm_help} Configure DashScope credentials in your ClickZetta connection first."
        )

    llm_enrichment = st.checkbox(
        "Use DashScope to enrich semantic metadata",
        help=llm_help,
        value=llm_available,
        disabled=not llm_available,
    )
    if not llm_available:
        llm_enrichment = False
    st.session_state["llm_enrichment"] = llm_enrichment

    submit = st.button("Submit", use_container_width=True, type="primary")
    if submit:
        try:
            run_generate_model_str_from_clickzetta(
                model_name,
                sample_values,
                st.session_state["selected_tables"],
                allow_joins=experimental_features,
                enrich_with_llm=llm_enrichment,
            )
            st.session_state["page"] = GeneratorAppScreen.ITERATION
            st.rerun()
        except ValueError as e:
            st.error(e)


def show() -> None:
    table_selector_dialog()
