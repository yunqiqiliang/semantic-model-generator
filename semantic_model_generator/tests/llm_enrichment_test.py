import json

from semantic_model_generator import generate_model
from semantic_model_generator.data_processing.data_types import Column, FQNParts, Table
from semantic_model_generator.llm.dashscope_client import DashscopeResponse
from semantic_model_generator.llm.enrichment import enrich_semantic_model
from semantic_model_generator.protos import semantic_model_pb2


class _FakeDashscopeClient:
    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def chat_completion(self, messages):  # type: ignore[no-untyped-def]
        return DashscopeResponse(content=json.dumps(self._payload, ensure_ascii=False), request_id="test")


def test_enrich_semantic_model_populates_descriptions_and_synonyms() -> None:
    raw_table = Table(
        id_=0,
        name="orders",
        columns=[
            Column(id_=0, column_name="order_status", column_type="STRING", values=["OPEN", "CLOSED"]),
            Column(id_=1, column_name="total_amount", column_type="NUMBER", values=["12.5", "18.3"]),
        ],
    )

    table_proto = semantic_model_pb2.Table(
        name="ORDERS",
        description="  ",
        base_table=semantic_model_pb2.FullyQualifiedTable(database="SALES", schema="PUBLIC", table="ORDERS"),
        dimensions=[
            semantic_model_pb2.Dimension(
                name="order_status",
                expr="order_status",
                data_type="STRING",
                description="  ",
                synonyms=["  "],
                sample_values=["OPEN", "CLOSED"],
            )
        ],
        time_dimensions=[
            semantic_model_pb2.TimeDimension(
                name="order_date",
                expr="order_date",
                data_type="TIMESTAMP_NTZ",
                description="  ",
                sample_values=["2024-02-15", "2024-03-10"],
            )
        ],
        facts=[
            semantic_model_pb2.Fact(
                name="total_amount",
                expr="total_amount",
                data_type="DECIMAL",
                description="  ",
                synonyms=["  "],
                sample_values=["12.5", "18.3"],
            )
        ],
        filters=[
            semantic_model_pb2.NamedFilter(
                name="order_status_include_values",
                expr="order_status IN ('OPEN', 'CLOSED')",
                description="  ",
                synonyms=["  "],
            )
        ],
    )

    model = semantic_model_pb2.SemanticModel(name="test", tables=[table_proto])

    fake_response = {
        "table_description": "Orders fact table that records order status and total amount.",
        "columns": [
            {
                "name": "order_status",
                "description": "Current execution status for each order.",
                "synonyms": ["Order status", "Fulfillment state"],
            },
            {
                "name": "total_amount",
                "description": "Order total including taxes.",
                "synonyms": ["Order amount", "Order total"],
            },
        ],
        "business_metrics": [
            {
                "name": "GMV",
                "source_columns": ["total_amount"],
                "description": "Based on total_amount and used as gross merchandise value.",
            }
        ],
        "filters": [
        {
            "name": "order_status_include_values",
            "description": "Limit the result set to a sample of order statuses.",
            "synonyms": ["Order status filter"],
        }
        ],
        "model_description": "Semantic model for customer orders and related metrics.",
    }

    client = _FakeDashscopeClient(fake_response)
    enrich_semantic_model(
        model,
        [(FQNParts(database="SALES", schema_name="PUBLIC", table="ORDERS"), raw_table)],
        client,
        placeholder="  ",
    )

    table = model.tables[0]
    assert table.description == "Orders fact table that records order status and total amount."

    dimension = next(dim for dim in table.dimensions if dim.expr == "order_status")
    assert dimension.description == "Current execution status for each order."
    assert "Order status" in list(dimension.synonyms)

    fact = next(f for f in table.facts if f.expr == "total_amount")
    assert fact.description == "Order total including taxes."
    assert "Order total" in list(fact.synonyms)

    filter_obj = next(flt for flt in table.filters if flt.name == "order_status_include_values")
    assert filter_obj.description == "Limit the result set to a sample of order statuses."
    assert "Order status filter" in list(filter_obj.synonyms)

    assert len(table.metrics) == 1
    metric = table.metrics[0]
    assert metric.name.startswith("gmv")
    assert metric.expr == "SUM(total_amount)"
    assert "GMV" in list(metric.synonyms)
    assert metric.description == "Based on total_amount and used as gross merchandise value."

    assert model.custom_instructions == ""
    assert model.description == "Semantic model for customer orders and related metrics."
