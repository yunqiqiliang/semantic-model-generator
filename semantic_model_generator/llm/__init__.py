from __future__ import annotations

from typing import Optional

from semantic_model_generator.clickzetta_utils import env_vars

from .dashscope_client import DashscopeClient, DashscopeSettings
from .enrichment import enrich_semantic_model

__all__ = [
    "DashscopeClient",
    "DashscopeSettings",
    "enrich_semantic_model",
    "get_dashscope_settings",
    "is_llm_available",
]


def get_dashscope_settings() -> Optional[DashscopeSettings]:
    api_key = env_vars.DASHSCOPE_API_KEY.strip()
    model = env_vars.DASHSCOPE_MODEL.strip()
    if not api_key or not model:
        return None
    return DashscopeSettings(
        api_key=api_key,
        base_url=env_vars.DASHSCOPE_BASE_URL.strip(),
        model=model,
        temperature=env_vars.DASHSCOPE_TEMPERATURE,
        top_p=env_vars.DASHSCOPE_TOP_P,
        max_output_tokens=env_vars.DASHSCOPE_MAX_OUTPUT_TOKENS,
        timeout_seconds=env_vars.DASHSCOPE_TIMEOUT_SECONDS,
    )


def is_llm_available() -> bool:
    return get_dashscope_settings() is not None
