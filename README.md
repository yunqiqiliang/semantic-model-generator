# semantic-model-generator

The ClickZetta Semantic Model Generator is a Streamlit-based companion app that helps teams inspect ClickZetta tables, scaffold semantic YAML, and iterate on partner integrations. It now targets ClickZetta’s Lakehouse APIs and ClickZetta volumes by default.

## Requirements

- Python 3.11
- Access to a ClickZetta workspace (service URL, instance, workspace, schema, vcluster, username, password)
- A `connections.json` file in one of the standard ClickZetta locations (`~/.clickzetta/connections.json`, `config/connections.json`, `config/lakehouse_connection/connections.json`, or `/app/.clickzetta/lakehouse_connection/connections.json`). The structure matches the template from [`mcp-clickzetta-server`](https://github.com/yunqiqiliang/mcp-clickzetta-server/blob/main/config/connections-template.json). Set `"is_default": true` for the connection the app should use.

```json
{
  "connections": [
    {
      "connection_name": "dev",
      "is_default": true,
      "service": "cn-shanghai-alicloud.api.clickzetta.com",
      "instance": "your_instance",
      "workspace": "quick_start",
      "schema": "PUBLIC",
      "username": "user",
      "password": "password",
      "vcluster": "default_ap"
    }
  ]
}
```

Environment variables such as `CLICKZETTA_SERVICE`, `CLICKZETTA_USERNAME`, etc. override the JSON values when present.

## Installation

```bash
# optional: conda env using environment.yml
conda env create -f environment.yml
conda activate clickzetta_env

# or install via poetry/pip
poetry install
# pip install .
```

The app depends on `clickzetta-connector-python` and `clickzetta-zettapark-python`; ensure they are installed via the commands above.

## Running the Streamlit app

```bash
# inside the Poetry environment
poetry run streamlit run app.py

# or, after activating the env, run:
python -m streamlit run app.py
```

When the app launches it will:

1. Load credentials from the ClickZetta connection config or environment.
2. Default file operations to `volume:user://~/semantic_models/` inside your user volume.
3. Provide workflows for generating semantic YAML, editing YAML, validating (basic checks), and importing partner specs (dbt, etc.).

## DashScope 使用提示

- 语义补全调用 DashScope 官方 SDK 默认端点，无需也无法通过 `base_url` 重写。
- 即便在 `connections.json` 或环境变量里设置 `DASHSCOPE_BASE_URL`/兼容端点，应用也不会使用这些值。
- 仍需提供 `DASHSCOPE_API_KEY` 与模型名称（如 `qwen-plus`）；其他参数保持默认即可避免常见 `InvalidParameter: url error`。
- 仅当你明确需要 OpenAI 兼容模式时才应使用兼容端点；当前 Streamlit 应用未对兼容端点提供支持。

## Key behaviours

- **Volume-first uploads**: YAML import/export uses the user volume path `volume:user://~/semantic_models/` unless a different volume/stage is selected.
- **Metadata discovery**: Workspace metadata (catalogs, schemas, tables) is fetched via ClickZetta INFORMATION_SCHEMA queries. Sample values and comments are collected using ClickZetta sessions.
- **Partner integrations**: dbt helpers read YAML from the chosen volume/stage, merge metadata, and reuse ClickZetta credentials.
- **Chat/validation placeholders**: Cortex-specific validation and chat calls are not yet available in ClickZetta mode; the UI will display placeholders instead of calling external services.

## Development scripts

Useful commands while iterating:

```bash
make setup        # install dependencies
make run_admin_app
make fmt_lint     # format + lint
make test         # execute pytest suite
```

## License

Apache 2.0 / BSD (dual license) – see LICENSE and LEGAL files for details.
