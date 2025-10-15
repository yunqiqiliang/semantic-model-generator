from __future__ import annotations

import json
import os
from dataclasses import dataclass
from http import HTTPStatus
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

from loguru import logger
try:
    import dashscope  # type: ignore
    from dashscope import Generation  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    dashscope = None  # type: ignore
    Generation = None  # type: ignore


@dataclass(frozen=True)
class DashscopeSettings:
    api_key: str
    model: str
    base_url: str = ""
    temperature: float = 0.2
    top_p: float = 0.85
    max_output_tokens: int = 512
    timeout_seconds: float = 45.0


class DashscopeError(RuntimeError):
    """Raised when DashScope returns an error response."""


@dataclass(frozen=True)
class DashscopeResponse:
    content: str
    request_id: Optional[str] = None


def _normalize_base_url(raw: str) -> str:
    raw = (raw or "").strip()
    if not raw:
        return ""
    parsed = urlparse(raw if "://" in raw else f"https://{raw}")
    if not parsed.netloc:
        return ""
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    path = parsed.path or ""
    if "compatible-mode" in path:
        path = path.replace("compatible-mode", "api")
    if not path or path == "/":
        path = "/api/v1"
    if not path.endswith("/api/v1"):
        path = path.rstrip("/") + "/api/v1"
    return urlunparse((scheme, netloc, path, "", "", ""))


class DashscopeClient:
    def __init__(self, settings: DashscopeSettings) -> None:
        if dashscope is None or Generation is None:
            raise DashscopeError(
                "DashScope Python SDK is not installed. Please add `dashscope` to your environment."
            )
        self._settings = settings
        self._normalized_base_url = _normalize_base_url(settings.base_url)

    def chat_completion(self, messages: List[Dict[str, str]]) -> DashscopeResponse:
        if dashscope is None or Generation is None:  # pragma: no cover - double guard
            raise DashscopeError(
                "DashScope Python SDK is not installed. Please add `dashscope` to your environment."
            )

        dashscope.api_key = self._settings.api_key
        os.environ.setdefault("DASHSCOPE_API_KEY", self._settings.api_key)

        response = None
        managed_env_keys = [
            "DASHSCOPE_API_BASE",
            "DASHSCOPE_BASE_URL",
            "DASHSCOPE_COMPAT_URL",
        ]
        previous_env: Dict[str, Optional[str]] = {}
        try:
            for key in managed_env_keys:
                previous_env[key] = os.environ.pop(key, None)

            normalized_base_url = self._normalized_base_url
            if normalized_base_url:
                os.environ["DASHSCOPE_BASE_URL"] = normalized_base_url
                if hasattr(dashscope, "base_http_api_url"):
                    dashscope.base_http_api_url = normalized_base_url  # type: ignore[attr-defined]

            model_name = self._settings.model or "qwen-plus"
            dashscope_model: Any = model_name
            if isinstance(model_name, str):
                normalized = model_name.strip().lower().replace("-", "_")
                if normalized.endswith("_latest"):
                    normalized = normalized[: -len("_latest")]
                if "embedding" in normalized:
                    normalized = "qwen_plus"
                if hasattr(Generation.Models, normalized):
                    dashscope_model = getattr(Generation.Models, normalized)
                elif "qwen" in normalized:
                    dashscope_model = "qwen-plus"
                else:
                    dashscope_model = Generation.Models.qwen_plus

            response = Generation.call(
                model=dashscope_model,
                messages=messages,
                stream=False,
                result_format="message",
                temperature=self._settings.temperature,
                top_p=self._settings.top_p,
                max_output_tokens=self._settings.max_output_tokens,
                timeout=self._settings.timeout_seconds,
            )
        except Exception as exc:  # pragma: no cover - SDK raised error
            raise DashscopeError(f"DashScope request failed: {exc}") from exc
        finally:
            for key, value in previous_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

        if response is None:
            raise DashscopeError("DashScope call did not return a response.")

        status = getattr(response, "status_code", None)
        if status != HTTPStatus.OK:
            error_message = getattr(response, "message", "Unknown error")
            error_code = getattr(response, "code", None)
            logger.error(
                "DashScope returned error ({}, code={}): {}",
                status,
                error_code,
                error_message,
            )
            raise DashscopeError(
                f"DashScope error {status} (code={error_code}): {error_message}"
            )

        output = getattr(response, "output", None)
        if not output or not hasattr(output, "choices"):
            raise DashscopeError(f"DashScope response missing output choices: {response}")

        choices = getattr(output, "choices")
        if not choices:
            raise DashscopeError("DashScope response contained no choices.")

        first = choices[0]
        message = getattr(first, "message", None)
        if not message or not hasattr(message, "content"):
            raise DashscopeError("DashScope response missing message content.")

        content = getattr(message, "content")
        if not content:
            raise DashscopeError("DashScope response returned empty content.")

        request_id = getattr(response, "request_id", None)
        return DashscopeResponse(content=content, request_id=request_id)
