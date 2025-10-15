from semantic_model_generator import generate_model
from semantic_model_generator.data_processing.data_types import Column, FQNParts, Table


def test_suggest_filters_builds_in_clause_and_time_filter() -> None:
    raw_table = Table(
        id_=0,
        name="ORDERS",
        columns=[
            Column(id_=0, column_name="status", column_type="STRING", values=["OPEN", "CLOSED", "OPEN"]),
            Column(id_=1, column_name="order_date", column_type="TIMESTAMP", values=["2024-01-01", "2024-01-15"]),
            Column(id_=2, column_name="created_at", column_type="STRING", values=["2024-01-05 10:00:00"]),
        ],
    )

    filters = generate_model._suggest_filters(raw_table)

    expressions = [flt.expr for flt in filters]
    assert any("IN (" in expr and "status" in expr for expr in expressions)
    assert any("DATEADD" in expr and "order_date" in expr for expr in expressions)
    assert any("DATEADD" in expr and "created_at" in expr for expr in expressions)


def test_infer_relationships_uses_pk_candidate() -> None:
    customers_table = Table(
        id_=0,
        name="CUSTOMERS",
        columns=[
            Column(id_=0, column_name="customer_id", column_type="INT", values=["1", "2", "3"]),
            Column(id_=1, column_name="customer_name", column_type="STRING"),
        ],
    )
    orders_table = Table(
        id_=1,
        name="ORDERS",
        columns=[
            Column(id_=0, column_name="order_id", column_type="INT", values=["10", "11", "12"]),
            Column(id_=1, column_name="customer_id", column_type="INT", values=["1", "2", "1"]),
        ],
    )

    relationships = generate_model._infer_relationships(
        [
            (FQNParts(database="QUICK_START", schema_name="MCP_DEMO", table="CUSTOMERS"), customers_table),
            (FQNParts(database="QUICK_START", schema_name="MCP_DEMO", table="ORDERS"), orders_table),
        ]
    )

    assert relationships
    relationship = relationships[0]
    assert relationship.left_table == "ORDERS"
    assert relationship.right_table == "CUSTOMERS"
    assert relationship.relationship_columns
    key = relationship.relationship_columns[0]
    assert key.left_column == "customer_id"
    assert key.right_column == "customer_id"


def test_infer_relationships_matches_synonym_keys() -> None:
    orders_table = Table(
        id_=0,
        name="ORDERS",
        columns=[
            Column(id_=0, column_name="o_orderkey", column_type="INT", values=["1", "2", "3"]),
            Column(id_=1, column_name="o_custkey", column_type="INT", values=["10", "20", "30"]),
        ],
    )
    lineitem_table = Table(
        id_=1,
        name="LINEITEM",
        columns=[
            Column(id_=0, column_name="l_orderkey", column_type="INT", values=["1", "1", "2"]),
            Column(id_=1, column_name="l_linenumber", column_type="INT", values=["1", "2", "1"]),
        ],
    )

    relationships = generate_model._infer_relationships(
        [
            (FQNParts(database="CAT", schema_name="SCH", table="ORDERS"), orders_table),
            (FQNParts(database="CAT", schema_name="SCH", table="LINEITEM"), lineitem_table),
        ]
    )

    assert relationships
    rel = relationships[0]
    assert rel.left_table == "LINEITEM"
    assert rel.right_table == "ORDERS"
    join = rel.relationship_columns[0]
    assert join.left_column == "l_orderkey"
    assert join.right_column == "o_orderkey"


def test_infer_relationships_handles_part_supplier() -> None:
    part_table = Table(
        id_=0,
        name="PART",
        columns=[
            Column(id_=0, column_name="p_partkey", column_type="INT", values=["1", "2", "3"]),
        ],
    )
    partsupp_table = Table(
        id_=1,
        name="PARTSUPP",
        columns=[
            Column(id_=0, column_name="ps_partkey", column_type="INT", values=["1", "1", "2"]),
            Column(id_=1, column_name="ps_suppkey", column_type="INT", values=["10", "20", "30"]),
        ],
    )

    relationships = generate_model._infer_relationships(
        [
            (FQNParts(database="CAT", schema_name="SCH", table="PART"), part_table),
            (FQNParts(database="CAT", schema_name="SCH", table="PARTSUPP"), partsupp_table),
        ]
    )

    assert relationships
    rel = relationships[0]
    assert rel.left_table == "PARTSUPP"
    assert rel.right_table == "PART"
    join = rel.relationship_columns[0]
    assert join.left_column == "ps_partkey"
    assert join.right_column == "p_partkey"


def test_infer_relationships_orders_customer() -> None:
    orders_table = Table(
        id_=0,
        name="ORDERS",
        columns=[
            Column(id_=0, column_name="o_orderkey", column_type="INT", values=["1", "2", "3"]),
            Column(id_=1, column_name="o_custkey", column_type="INT", values=["10", "20", "30"]),
        ],
    )
    customer_table = Table(
        id_=1,
        name="CUSTOMER",
        columns=[
            Column(id_=0, column_name="c_custkey", column_type="INT", values=["10", "20", "30"]),
            Column(id_=1, column_name="c_name", column_type="STRING"),
        ],
    )

    relationships = generate_model._infer_relationships(
        [
            (FQNParts(database="CAT", schema_name="SCH", table="ORDERS"), orders_table),
            (FQNParts(database="CAT", schema_name="SCH", table="CUSTOMER"), customer_table),
        ]
    )

    assert relationships
    rel = relationships[0]
    assert rel.left_table == "ORDERS"
    assert rel.right_table == "CUSTOMER"
    join = rel.relationship_columns[0]
    assert join.left_column == "o_custkey"
    assert join.right_column == "c_custkey"


def test_infer_relationships_lineitem_supplier() -> None:
    lineitem_table = Table(
        id_=0,
        name="LINEITEM",
        columns=[
            Column(id_=0, column_name="l_orderkey", column_type="INT", values=["1", "2", "3"]),
            Column(id_=1, column_name="l_suppkey", column_type="INT", values=["100", "101", "102"]),
        ],
    )
    supplier_table = Table(
        id_=1,
        name="SUPPLIER",
        columns=[
            Column(id_=0, column_name="s_suppkey", column_type="INT", values=["100", "101", "102"]),
            Column(id_=1, column_name="s_name", column_type="STRING"),
        ],
    )

    relationships = generate_model._infer_relationships(
        [
            (FQNParts(database="CAT", schema_name="SCH", table="LINEITEM"), lineitem_table),
            (FQNParts(database="CAT", schema_name="SCH", table="SUPPLIER"), supplier_table),
        ]
    )

    assert relationships
    rel = relationships[0]
    assert rel.left_table == "LINEITEM"
    assert rel.right_table == "SUPPLIER"
    join = rel.relationship_columns[0]
    assert join.left_column == "l_suppkey"
    assert join.right_column == "s_suppkey"


def test_infer_relationships_handles_suffix_based_foreign_keys() -> None:
    dim_date = Table(
        id_=0,
        name="DIM_DATE",
        columns=[
            Column(id_=0, column_name="date_id", column_type="INT", values=["20240101", "20240102"], is_primary_key=True),
            Column(id_=1, column_name="date_value", column_type="DATE"),
        ],
    )
    fact_sales = Table(
        id_=1,
        name="FACT_SALES",
        columns=[
            Column(id_=0, column_name="order_id", column_type="INT", values=["10", "11"]),
            Column(id_=1, column_name="order_date_id", column_type="INT", values=["20240101", "20240102"]),
        ],
    )

    relationships = generate_model._infer_relationships(
        [
            (FQNParts(database="CAT", schema_name="SCH", table="DIM_DATE"), dim_date),
            (FQNParts(database="CAT", schema_name="SCH", table="FACT_SALES"), fact_sales),
        ]
    )

    assert relationships
    rel = relationships[0]
    assert rel.left_table == "FACT_SALES"
    assert rel.right_table == "DIM_DATE"
    join = rel.relationship_columns[0]
    assert join.left_column == "order_date_id"
    assert join.right_column == "date_id"
