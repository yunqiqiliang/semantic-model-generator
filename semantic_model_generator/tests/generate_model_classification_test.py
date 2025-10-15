from semantic_model_generator import generate_model
from semantic_model_generator.data_processing.data_types import Column, Table


def test_identifier_like_numeric_stays_dimension() -> None:
    raw_table = Table(
        id_=0,
        name="CUSTOMERS",
        columns=[
            Column(id_=0, column_name="customer_id", column_type="INT"),
            Column(id_=1, column_name="loyalty_score", column_type="NUMBER"),
        ],
    )

    table = generate_model._raw_table_to_semantic_context_table(
        database="QUICK_START",
        schema="MCP_DEMO",
        raw_table=raw_table,
    )

    dimension_names = {dim.name for dim in table.dimensions}
    fact_names = {fact.name for fact in table.facts}

    assert "customer_id" in dimension_names
    assert "customer_id" not in fact_names
    assert "loyalty_score" in fact_names


def test_string_date_promoted_to_time_dimension() -> None:
    raw_table = Table(
        id_=0,
        name="ORDERS",
        columns=[
            Column(id_=0, column_name="order_date", column_type="STRING", values=["2024-01-01", "2024-02-01"]),
            Column(id_=1, column_name="order_status", column_type="STRING", values=["OPEN", "CLOSED"]),
        ],
    )

    table = generate_model._raw_table_to_semantic_context_table(
        database="QUICK_START",
        schema="MCP_DEMO",
        raw_table=raw_table,
    )

    time_dimension_names = {td.name for td in table.time_dimensions}
    dimension_names = {dim.name for dim in table.dimensions}

    assert "order_date" in time_dimension_names
    assert "order_date" not in dimension_names
    time_dim = next(td for td in table.time_dimensions if td.name == "order_date")
    assert time_dim.data_type == "TIMESTAMP_NTZ"


def test_identifier_sanitization_and_primary_key_detection() -> None:
    assert generate_model._sanitize_identifier_name("o_orderkey") == "ORDERKEY"
    assert generate_model._sanitize_identifier_name("L_PartKey") == "PARTKEY"
    assert generate_model._looks_like_primary_key("ORDERS", "o_orderkey")
    assert generate_model._looks_like_primary_key("NATION", "n_nationkey")
