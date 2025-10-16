from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

from semantic_model_generator.data_processing import data_types
from semantic_model_generator.protos import semantic_model_pb2

from .dashscope_client import DashscopeClient, DashscopeError

_JSON_BLOCK_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_NUMERIC_TYPES = {
    "NUMBER",
    "DECIMAL",
    "INT",
    "INTEGER",
    "FLOAT",
    "DOUBLE",
    "BIGINT",
    "SMALLINT",
}
_NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Z]+")

SYSTEM_PROMPT = (
    "You are an experienced ClickZetta data analyst. "
    "Only respond in English. Write concise, professional descriptions that help analysts understand table purpose, column semantics, and business metrics. "
    "Preserve original column names when you reference them."
)


def enrich_semantic_model(
    model: semantic_model_pb2.SemanticModel,
    raw_tables: Sequence[Tuple[data_types.FQNParts, data_types.Table]],
    client: DashscopeClient,
    placeholder: str = "  ",
) -> None:
    """
    Enriches the semantic model in-place using DashScope generated descriptions.
    """

    if not model.tables or not raw_tables:
        return

    raw_lookup: Dict[str, data_types.Table] = {tbl.name.upper(): tbl for _, tbl in raw_tables}
    metric_notes: List[str] = []

    for table in model.tables:
        raw_table = raw_lookup.get(table.name.upper())
        if not raw_table:
            logger.debug("No raw metadata for table {}; skipping enrichment.", table.name)
            continue
        try:
            payload = _serialize_table_prompt(table, raw_table, model.description, placeholder)
            response = client.chat_completion(payload["messages"])
            enrichment = _parse_llm_response(response.content)
            if enrichment:
                updates = _apply_enrichment(table, raw_table, enrichment, placeholder)
                note = updates.get("business_notes")
                if note and not updates.get("metrics_added"):
                    metric_notes.append(f"{table.name}: {note}")
                model_description = updates.get("model_description")
                if (
                    model_description
                    and isinstance(model_description, str)
                    and (model.description == placeholder or not model.description.strip())
                ):
                    model.description = model_description.strip()
        except DashscopeError as exc:  # pragma: no cover - network failures or remote errors
            logger.warning("DashScope enrichment failed for {}: {}", table.name, exc)
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.exception("Unexpected error enriching table {}: {}", table.name, exc)
    if model.description == placeholder or not model.description.strip():
        _summarize_model_description(model, client, placeholder)

    if metric_notes:
        model.custom_instructions = "\n".join(metric_notes)
    else:
        model.custom_instructions = ""


def _serialize_table_prompt(
    table: semantic_model_pb2.Table,
    raw_table: data_types.Table,
    model_description: str,
    placeholder: str,
) -> Dict[str, Any]:
    column_roles: Dict[str, str] = {}
    column_descriptions: Dict[str, str] = {}

    for dim in table.dimensions:
        column_roles[dim.expr.upper()] = "dimension"
        column_descriptions[dim.expr.upper()] = dim.description
    for td in table.time_dimensions:
        column_roles[td.expr.upper()] = "time_dimension"
        column_descriptions[td.expr.upper()] = td.description
    for fact in table.facts:
        column_roles[fact.expr.upper()] = "fact"
        column_descriptions[fact.expr.upper()] = fact.description

    columns_payload: List[Dict[str, object]] = []
    for col in raw_table.columns:
        upper_name = col.column_name.upper()
        role = column_roles.get(upper_name, "unknown")
        description = column_descriptions.get(upper_name, "")
        if description == placeholder:
            description = ""
        columns_payload.append(
            {
                "name": col.column_name,
                "role": role,
                "data_type": col.column_type,
                "has_description": bool(description.strip()),
                "sample_values": col.values[:5] if col.values else [],
            }
        )

    prompt_payload = {
        "table_name": table.name,
        "table_has_description": table.description.strip() not in {placeholder, ""},
        "table_comment": raw_table.comment or "",
        "columns": columns_payload,
        "filters": [
            {
                "name": nf.name,
                "expr": nf.expr,
                "has_description": bool(nf.description.strip()),
                "has_synonyms": any(s.strip() and s != placeholder for s in nf.synonyms),
            }
            for nf in table.filters
        ],
        "semantic_model_description": model_description,
    }

    user_instructions = (
        "Review the JSON metadata below and reply with a strictly JSON response.\n"
        "1. If a table or column description is empty, provide a concise English description; do not duplicate existing text.\n"
        "2. For facts (numeric columns), propose business-friendly synonyms and explain what the metric represents.\n"
        "3. For dimensions and time dimensions, include common English aliases when useful.\n"
        "4. For filters, provide helpful descriptions and synonyms when they are missing.\n"
        "5. Optionally suggest up to two derived business metrics in a `business_metrics` list with `name`, `source_columns`, `description`, and optionally `synonyms`.\n"
        "6. Provide `model_description` if you can summarize how this table contributes to the overall semantic model.\n"
        "7. Keep column and filter names unchanged and respond with valid JSON only.\n\n"
        "Example output:\n"
        "{\n"
        '  "table_description": "Orders fact table that captures the status and finances of each order",\n'
        '  "columns": [\n'
        '    {\n'
        '      "name": "O_TOTALPRICE",\n'
        '      "description": "Total order value including tax",\n'
        '      "synonyms": ["Order amount", "Order total"]\n'
        "    }\n"
        "  ],\n"
        '  "business_metrics": [\n'
        '    {\n'
        '      "name": "Gross merchandise value",\n'
        '      "source_columns": ["O_TOTALPRICE"],\n'
        '      "description": "Used to measure GMV derived from the total order price."\n'
        "    }\n"
        "  ]\n"
        "}\n\n"
        f"Metadata: ```json\n{json.dumps(prompt_payload, ensure_ascii=False, indent=2)}\n```"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_instructions},
    ]
    return {"messages": messages}


def _parse_llm_response(content: str) -> Optional[Dict[str, object]]:
    if not content:
        return None
    match = _JSON_BLOCK_PATTERN.search(content)
    json_text = match.group(0) if match else content
    json_text = json_text.strip().strip("`")
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        logger.warning("Unable to parse DashScope response as JSON: {} | raw={}", exc, content)
        return None
    if not isinstance(data, dict):
        return None
    return data


def _apply_enrichment(
    table: semantic_model_pb2.Table,
    raw_table: data_types.Table,
    enrichment: Dict[str, object],
    placeholder: str,
) -> Dict[str, Optional[str]]:
    result: Dict[str, Optional[str]] = {
        "business_notes": None,
        "model_description": None,
        "metrics_added": False,
    }
    table_description = enrichment.get("table_description")
    if isinstance(table_description, str) and table.description == placeholder:
        table.description = table_description.strip()

    column_entries = enrichment.get("columns", [])
    if isinstance(column_entries, list):
        _apply_column_enrichment(table, column_entries, placeholder)

    business_metrics = enrichment.get("business_metrics")
    business_notes = None
    if isinstance(business_metrics, list) and business_metrics:
        business_notes, metrics_added = _apply_metric_enrichment(
            table, raw_table, business_metrics, placeholder
        )
        result["metrics_added"] = metrics_added
    _apply_filter_enrichment(table, enrichment, placeholder)
    result["business_notes"] = business_notes
    model_description = enrichment.get("model_description")
    if isinstance(model_description, str) and model_description.strip():
        result["model_description"] = model_description.strip()
    return result


def _apply_column_enrichment(
    table: semantic_model_pb2.Table,
    column_entries: Iterable[object],
    placeholder: str,
) -> None:
    dim_map = {dim.expr.upper(): dim for dim in table.dimensions}
    time_map = {td.expr.upper(): td for td in table.time_dimensions}
    fact_map = {fact.expr.upper(): fact for fact in table.facts}

    for entry in column_entries:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str):
            continue
        upper = name.upper()
        target = dim_map.get(upper) or time_map.get(upper) or fact_map.get(upper)
        if not target:
            continue

        description = entry.get("description")
        if isinstance(description, str) and getattr(target, "description", "") == placeholder:
            target.description = description.strip()

        synonyms = entry.get("synonyms")
        if isinstance(synonyms, list):
            _apply_synonyms(target, synonyms, placeholder)


def _apply_synonyms(target: object, synonyms: Sequence[object], placeholder: str) -> None:
    clean_synonyms: List[str] = []
    for item in synonyms:
        if isinstance(item, str):
            text = item.strip()
            if text:
                clean_synonyms.append(text)
    if not clean_synonyms:
        return

    existing = [syn for syn in getattr(target, "synonyms", []) if syn.strip() and syn != placeholder]
    merged = _deduplicate(existing + clean_synonyms)

    if hasattr(target, "synonyms"):
        container = getattr(target, "synonyms")
        del container[:]
        container.extend(merged)


def _deduplicate(values: Sequence[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        upper = value.upper()
        if upper in seen:
            continue
        seen.add(upper)
        result.append(value)
    return result


def _build_business_metric_notes(metrics: Sequence[object]) -> str:
    lines: List[str] = []
    for metric in metrics:
        if not isinstance(metric, dict):
            continue
        name = metric.get("name")
        description = metric.get("description")
        sources = metric.get("source_columns", [])
        if not isinstance(name, str) or not name.strip():
            continue
        detail_parts: List[str] = [name.strip()]
        if isinstance(sources, list):
            clean_sources = [str(src).strip() for src in sources if str(src).strip()]
            if clean_sources:
                detail_parts.append(f"(source columns: {', '.join(clean_sources)})")
        if isinstance(description, str) and description.strip():
            detail_parts.append(f"- {description.strip()}")
        lines.append(" ".join(detail_parts))

    return "\n".join(lines)


def _apply_filter_enrichment(
    table: semantic_model_pb2.Table,
    enrichment: Dict[str, object],
    placeholder: str,
) -> None:
    if "filters" not in enrichment:
        return
    filter_map = {nf.name: nf for nf in table.filters}
    filters = enrichment.get("filters", [])
    if not isinstance(filters, list):
        return
    for entry in filters:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        target = filter_map.get(name)
        if not target:
            continue
        description = entry.get("description")
        if isinstance(description, str) and target.description == placeholder:
            target.description = description.strip()
        synonyms = entry.get("synonyms")
        if isinstance(synonyms, list):
            clean_synonyms = [str(item).strip() for item in synonyms if isinstance(item, (str, int, float))]
            if clean_synonyms:
                del target.synonyms[:]
                target.synonyms.extend(clean_synonyms)


def _sanitize_metric_name(name: str, existing: set[str]) -> str:
    cleaned = _NON_ALNUM_RE.sub("_", name.strip().lower()).strip("_")
    if not cleaned:
        cleaned = "metric"
    if cleaned[0].isdigit():
        cleaned = f"metric_{cleaned}"
    candidate = cleaned
    counter = 2
    while candidate in existing:
        candidate = f"{cleaned}_{counter}"
        counter += 1
    existing.add(candidate)
    return candidate


_COUNT_KEYWORDS = (
    "count",
    "number of",
    "total number",
    "how many",
    "volume of",
    "frequency",
    "headcount",
)
_DISTINCT_KEYWORDS = ("distinct", "unique", "deduplicated")
_AVERAGE_KEYWORDS = ("average", "avg", "mean", "typical", "expected", "per order", "per customer")
_SUM_KEYWORDS = (
    "total",
    "sum",
    "revenue",
    "value",
    "amount",
    "volume",
    "sales",
    "inventory",
    "cost",
    "spend",
    "margin",
)
_PRODUCT_KEYWORDS = (
    "multiply",
    "multiplied",
    "times",
    "product of",
    "extended price",
    "net",
    "after discount",
    "inventory value",
    "combined value",
)


def _collect_metric_text(entry: Dict[str, object]) -> str:
    parts: List[str] = []
    for field in ("name", "description"):
        value = entry.get(field)
        if isinstance(value, str):
            parts.append(value)
    synonyms = entry.get("synonyms")
    if isinstance(synonyms, list):
        for syn in synonyms:
            if isinstance(syn, (str, int, float)):
                parts.append(str(syn))
    return " ".join(parts).lower()


def _is_numeric_type(column_type: str) -> bool:
    upper_type = (column_type or "").upper()
    return any(token in upper_type for token in _NUMERIC_TYPES)


def _derive_metric_intent(
    entry: Dict[str, object],
    source_columns: Sequence[str],
    column_type_map: Dict[str, str],
) -> Tuple[str, bool]:
    """
    Determine the preferred aggregation function and whether a product expression
    should be used when multiple source columns are present.
    """
    text = _collect_metric_text(entry)
    aggregation: Optional[str] = None

    if any(keyword in text for keyword in _AVERAGE_KEYWORDS):
        aggregation = "AVG"

    if aggregation is None and any(keyword in text for keyword in _COUNT_KEYWORDS):
        aggregation = "COUNT"
        if any(keyword in text for keyword in _DISTINCT_KEYWORDS):
            aggregation = "COUNT_DISTINCT"

    if aggregation is None and any(keyword in text for keyword in _SUM_KEYWORDS):
        aggregation = "SUM"

    if aggregation is None:
        aggregation = "SUM"

    use_product = False
    if (
        aggregation == "SUM"
        and len(source_columns) >= 2
        and any(keyword in text for keyword in _PRODUCT_KEYWORDS)
    ):
        first_type = column_type_map.get(source_columns[0].upper(), "")
        second_type = column_type_map.get(source_columns[1].upper(), "")
        if _is_numeric_type(first_type) and _is_numeric_type(second_type):
            use_product = True

    return aggregation, use_product


def _build_metric_expression(
    source_columns: Sequence[str],
    column_type_map: Dict[str, str],
    aggregation: str,
    use_product: bool,
) -> str:
    if not source_columns:
        raise ValueError("No source columns provided for metric enrichment.")

    column_name = source_columns[0]
    column_type = column_type_map.get(column_name.upper(), "")

    if aggregation == "COUNT_DISTINCT":
        return f"COUNT(DISTINCT {column_name})"

    if aggregation == "COUNT":
        return f"COUNT({column_name})"

    if aggregation == "AVG":
        if _is_numeric_type(column_type):
            return f"AVG({column_name})"
        # Fallback to COUNT if average is requested on a non-numeric column.
        return f"COUNT({column_name})"

    # SUM or default aggregation path.
    if use_product and len(source_columns) >= 2:
        return f"SUM({source_columns[0]} * {source_columns[1]})"

    if _is_numeric_type(column_type):
        return f"SUM({column_name})"

    # Fallback for non-numeric columns when SUM was requested.
    return f"COUNT({column_name})"


def _apply_metric_enrichment(
    table: semantic_model_pb2.Table,
    raw_table: data_types.Table,
    business_metrics: Sequence[object],
    placeholder: str,
) -> tuple[Optional[str], bool]:
    column_type_map = {col.column_name.upper(): col.column_type for col in raw_table.columns}
    existing_names: set[str] = {metric.name for metric in table.metrics}
    notes: List[Dict[str, object]] = []
    metrics_added = False

    for entry in business_metrics:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        raw_sources = entry.get("source_columns")
        resolved_sources: List[str] = []
        if isinstance(raw_sources, list):
            for col in raw_sources:
                if isinstance(col, str) and col.strip():
                    resolved_sources.append(col.strip())
        if not resolved_sources:
            if table.facts:
                resolved_sources = [table.facts[0].expr]
            else:
                continue

        metric_name = _sanitize_metric_name(name, existing_names)
        aggregation, use_product = _derive_metric_intent(entry, resolved_sources, column_type_map)
        expression = _build_metric_expression(resolved_sources, column_type_map, aggregation, use_product)

        metric = table.metrics.add()
        metric.name = metric_name
        metric.expr = expression

        description = entry.get("description")
        metric.description = (
            description.strip() if isinstance(description, str) and description.strip() else placeholder
        )

        synonyms = entry.get("synonyms")
        synonyms_list: List[str] = []
        if isinstance(synonyms, list):
            for syn in synonyms:
                if isinstance(syn, (str, int, float)):
                    text = str(syn).strip()
                    if text:
                        synonyms_list.append(text)
        if not synonyms_list:
            synonyms_list.append(name.strip())
        metric.synonyms.extend(synonyms_list)

        notes.append(
            {
                "name": name.strip(),
                "source_columns": raw_sources if isinstance(raw_sources, list) and raw_sources else resolved_sources,
                "description": description.strip() if isinstance(description, str) and description.strip() else "",
            }
        )
        metrics_added = True

    if notes:
        return _build_business_metric_notes(notes), metrics_added
    return None, metrics_added


def _summarize_model_description(
    model: semantic_model_pb2.SemanticModel,
    client: DashscopeClient,
    placeholder: str,
) -> None:
    if model.description != placeholder and model.description.strip():
        return

    table_lines = []
    for table in model.tables:
        role = "fact" if table.facts or table.metrics else "dimension"
        desc = table.description.strip() if table.description.strip() else "No description"
        metrics = ", ".join(metric.name for metric in table.metrics) or "None"
        table_lines.append(f"- {table.name} ({role}): {desc}. Metrics: {metrics}")

    relationship_lines = []
    for rel in model.relationships:
        parts = [f"{rel.left_table} -> {rel.right_table}"]
        if rel.relationship_columns:
            columns = ", ".join(
                f"{col.left_column}={col.right_column}" for col in rel.relationship_columns
            )
            parts.append(f"on {columns}")
        relationship_lines.append(" ".join(parts))
    if not relationship_lines:
        relationship_lines.append("No relationships provided")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a data modeling assistant. Given the semantic model information, "
                "write one or two concise English sentences that describe the overall purpose of the model "
                "and how its tables relate."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Semantic model name: {model.name}\n"
                f"Tables:\n{chr(10).join(table_lines)}\n"
                f"Relationships:\n{chr(10).join(relationship_lines)}"
            ),
        },
    ]

    try:
        response = client.chat_completion(messages)
        summary = response.content.strip()
        if summary:
            model.description = summary
    except DashscopeError as exc:  # pragma: no cover - defensive
        logger.warning("Failed to summarize semantic model description: {}", exc)
