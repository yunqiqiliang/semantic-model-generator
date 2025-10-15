from typing import Any, Dict

import streamlit as st

import json
from typing import Any, Dict, List

import streamlit as st

from semantic_model_generator.clickzetta_utils.clickzetta_connector import (
    ClickzettaConnectionProxy,
)
from semantic_model_generator.llm import DashscopeClient, get_dashscope_settings
from semantic_model_generator.llm.dashscope_client import DashscopeError


@st.cache_data(ttl=60, show_spinner=False)
def send_message(
    _conn: ClickzettaConnectionProxy,
    semantic_model: str,
    messages: list[dict[str, str]],
) -> Dict[str, Any]:
    """
    Uses DashScope's Qwen model to respond to analyst-style questions in English.
    """

    _ = _conn  # reserved for future transactional integrations

    settings = get_dashscope_settings()
    if not settings:
        raise ValueError(
            "DashScope credentials are not configured. Update your connections.json or environment variables."
        )
    if settings.model != "qwen-plus-latest":
        settings = type(settings)(
            api_key=settings.api_key,
            base_url=settings.base_url,
            model="qwen-plus-latest",
            temperature=settings.temperature,
            top_p=settings.top_p,
            max_output_tokens=settings.max_output_tokens,
            timeout_seconds=settings.timeout_seconds,
        )

    def _flatten(message: Dict[str, Any]) -> str:
        parts: List[str] = []
        for item in message.get("content", []):
            if item.get("type") == "text":
                parts.append(item.get("text", ""))
            elif item.get("type") == "sql":
                parts.append(f"SQL:\n{item.get('statement', '')}")
            elif item.get("type") == "suggestions":
                suggestions = item.get("suggestions", [])
                if suggestions:
                    parts.append("Suggestions: " + ", ".join(map(str, suggestions)))
        return "\n\n".join(filter(None, parts))

    instruction = (
        "You are ClickZetta Analyst, an English-only assistant that answers questions using the provided semantic model. "
        "Whenever you reference a table in SQL, fully qualify it using the exact database.schema.table shown in the semantic model YAML. "
        "Alias tables as needed, but never omit the database or schema. "
        "Respond with a JSON object containing:\n"
        '  {"analysis": "<concise English explanation>", "sql": "<optional SQL with fully-qualified tables>", "suggestions": ["optional follow-up questions"]}\n'
        "Only include keys that have values. If no SQL is appropriate, omit the field."
    )

    llm_messages: List[Dict[str, str]] = [
        {"role": "system", "content": instruction},
        {
            "role": "system",
            "content": f"Semantic model YAML:\n{semantic_model}",
        },
    ]

    role_map = {"analyst": "assistant"}
    for message in messages:
        role = role_map.get(message.get("role", "user"), message.get("role", "user"))
        llm_messages.append({"role": role, "content": _flatten(message)})

    try:
        client = DashscopeClient(settings)
        response = client.chat_completion(llm_messages)
    except DashscopeError as exc:
        raise ValueError(str(exc)) from exc

    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError:
        analysis = response.content.strip()
        sql_text = None
        suggestions: List[str] = []
    else:
        analysis = str(payload.get("analysis", "")).strip()
        sql_text = payload.get("sql")
        if isinstance(sql_text, str):
            sql_text = sql_text.strip()
        else:
            sql_text = None
        suggestions = [
            str(item).strip()
            for item in payload.get("suggestions", [])
            if isinstance(item, (str, int, float))
        ]

    if not analysis:
        analysis = "I'm sorry, I could not generate an answer based on the current semantic model."

    content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": analysis,
        }
    ]
    if sql_text:
        content.append({"type": "sql", "statement": sql_text})
    if suggestions:
        content.append({"type": "suggestions", "suggestions": suggestions})

    return {
        "message": {"content": content},
        "request_id": response.request_id or "dashscope",
    }
