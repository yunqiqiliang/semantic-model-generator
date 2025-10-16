import json
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(override=True)

_CONFIG_PATHS = [
    "/app/.clickzetta/lakehouse_connection/connections.json",
    "/app/config/lakehouse_connection/connections.json",
    "config/connections.json",
    "config/lakehouse_connection/connections.json",
    os.path.expanduser("~/.clickzetta/connections.json"),
    "/app/.clickzetta/connections.json",
]


_ACTIVE_CONFIG_PATH: Optional[str] = None


def _load_config_from_file() -> Tuple[Optional[Dict[str, str]], Dict[str, Dict[str, str]]]:
    global _ACTIVE_CONFIG_PATH
    _ACTIVE_CONFIG_PATH = None
    for path in _CONFIG_PATHS:
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue

        connections = data.get("connections", [])
        selected: Optional[Dict[str, str]] = None
        for conn in connections:
            if conn.get("is_default"):
                selected = conn
                break
        if not selected and connections:
            selected = connections[0]

        if selected:
            _ACTIVE_CONFIG_PATH = path
            return selected, data.get("system_config", {})
    return None, {}


CONFIG_CONNECTION, CONFIG_SYSTEM = _load_config_from_file()
ACTIVE_CONFIG_PATH = _ACTIVE_CONFIG_PATH
if not isinstance(CONFIG_SYSTEM, dict):
    CONFIG_SYSTEM = {}

_DASHSCOPE_CONFIG = {}
if CONFIG_SYSTEM:
    candidate = CONFIG_SYSTEM.get("dashscope")
    if isinstance(candidate, dict):
        _DASHSCOPE_CONFIG = candidate
    else:
        embedding_cfg = CONFIG_SYSTEM.get("embedding")
        if isinstance(embedding_cfg, dict):
            dashscope_cfg = embedding_cfg.get("dashscope")
            if isinstance(dashscope_cfg, dict):
                _DASHSCOPE_CONFIG = dashscope_cfg.copy()

if not isinstance(_DASHSCOPE_CONFIG, dict):
    _DASHSCOPE_CONFIG = {}


def _deep_lookup(mapping: Any, key: str) -> Optional[Any]:
    """
    Recursively searches dictionaries/lists for the first occurrence of ``key`` (case-insensitive).
    Returns the associated value if it is not empty/None.
    """

    if not isinstance(mapping, (dict, list)):
        return None

    normalized_key = key.lower()
    queue: deque[Any] = deque([mapping])
    seen: set[int] = set()

    while queue:
        current = queue.popleft()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)

        if isinstance(current, dict):
            for candidate_key, candidate_value in current.items():
                candidate_key_str = str(candidate_key).lower()
                if candidate_key_str == normalized_key and candidate_value not in (None, ""):
                    return candidate_value
                if isinstance(candidate_value, (dict, list)):
                    queue.append(candidate_value)
        elif isinstance(current, list):
            for item in current:
                if isinstance(item, (dict, list)):
                    queue.append(item)
    return None


def _config_value(key: str, default: Optional[str] = None) -> Optional[str]:
    env_value = os.getenv(f"CLICKZETTA_{key.upper()}")
    if env_value:
        return env_value
    if CONFIG_CONNECTION:
        value = CONFIG_CONNECTION.get(key)
        if value in ("", None):
            value = _deep_lookup(CONFIG_CONNECTION, key)
        if value not in ("", None):
            return str(value)
    return default


_session_cfg = CONFIG_SYSTEM.get("session", {}) if CONFIG_SYSTEM else {}
if not isinstance(_session_cfg, dict):
    _session_cfg = {}
DEFAULT_SESSION_TIMEOUT_SEC = int(
    os.environ.get("CLICKZETTA_SESSION_TIMEOUT_SEC")
    or _session_cfg.get("timeout_seconds")
    or 300
)

CLICKZETTA_SERVICE = _config_value("service")
CLICKZETTA_INSTANCE = _config_value("instance")
CLICKZETTA_WORKSPACE = _config_value("workspace")
CLICKZETTA_SCHEMA = _config_value("schema")
CLICKZETTA_USERNAME = _config_value("username")
CLICKZETTA_PASSWORD = _config_value("password")
CLICKZETTA_VCLUSTER = _config_value("vcluster", "default_ap")
CLICKZETTA_QUERY_TAG = _config_value("query_tag", "semantic-model-generator")
CLICKZETTA_HINTS: Dict[str, str] = {}
if CONFIG_CONNECTION and isinstance(CONFIG_CONNECTION.get("hints"), dict):
    CLICKZETTA_HINTS = {str(k): str(v) for k, v in CONFIG_CONNECTION["hints"].items()}


def build_base_connection_config() -> Dict[str, str]:
    """
    Returns the base configuration dictionary used to establish Clickzetta sessions.
    """

    config: Dict[str, str] = {
        "service": CLICKZETTA_SERVICE or "",
        "instance": CLICKZETTA_INSTANCE or "",
        "workspace": CLICKZETTA_WORKSPACE or "",
        "schema": CLICKZETTA_SCHEMA or "",
        "username": CLICKZETTA_USERNAME or "",
        "password": CLICKZETTA_PASSWORD or "",
        "vcluster": CLICKZETTA_VCLUSTER or "default_ap",
        "hints": CLICKZETTA_HINTS.copy(),
    }
    return config


def assert_required_env_vars() -> List[str]:
    """
    Ensures that the required environment variables are set before proceeding.

    Returns:
        List[str]: Names of missing environment variables.
    """

    missing_env_vars: List[str] = []
    if not CLICKZETTA_SERVICE:
        missing_env_vars.append("CLICKZETTA_SERVICE")
    if not CLICKZETTA_INSTANCE:
        missing_env_vars.append("CLICKZETTA_INSTANCE")
    if not CLICKZETTA_WORKSPACE:
        missing_env_vars.append("CLICKZETTA_WORKSPACE")
    if not CLICKZETTA_SCHEMA:
        missing_env_vars.append("CLICKZETTA_SCHEMA")
    if not CLICKZETTA_USERNAME:
        missing_env_vars.append("CLICKZETTA_USERNAME")
    if not CLICKZETTA_PASSWORD:
        missing_env_vars.append("CLICKZETTA_PASSWORD")

    return missing_env_vars


def _dashscope_value(key: str, env_key: Optional[str] = None) -> Optional[str]:
    lookup_key = env_key or f"DASHSCOPE_{key.upper()}"
    env_value = os.getenv(lookup_key)
    if env_value:
        return env_value
    value = _DASHSCOPE_CONFIG.get(key)
    if value is None:
        aliases = {
            "base_url": ["api_base", "base_url"],
            "api_key": ["api_key"],
            "model": ["model"],
        }.get(key, [])
        for alias in aliases:
            if alias == key:
                continue
            candidate = _DASHSCOPE_CONFIG.get(alias)
            if candidate is not None:
                value = candidate
                break
    if value is not None:
        return str(value)
    return None


def _dashscope_float_value(key: str, default: float) -> float:
    value = _dashscope_value(key)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _dashscope_int_value(key: str, default: int) -> int:
    value = _dashscope_value(key)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


DASHSCOPE_API_KEY = _dashscope_value("api_key") or ""
DASHSCOPE_MODEL = _dashscope_value("model") or "qwen-plus-latest"
DASHSCOPE_BASE_URL = _dashscope_value("base_url") or ""
DASHSCOPE_TEMPERATURE = _dashscope_float_value("temperature", 0.2)
DASHSCOPE_TOP_P = _dashscope_float_value("top_p", 0.85)
DASHSCOPE_MAX_OUTPUT_TOKENS = _dashscope_int_value("max_output_tokens", 512)
DASHSCOPE_TIMEOUT_SECONDS = _dashscope_float_value("timeout_seconds", 45.0)
