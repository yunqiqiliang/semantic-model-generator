# Repository Guidelines

## Project Structure & Module Ownership
`app.py` launches the Streamlit UI. Shared UI state, messaging, and storage logic live in `app_utils/`. Domain code is grouped under `semantic_model_generator/`: `clickzetta_utils/` handles credentials, sessions, and SHOW/LIST helpers; `data_processing/` transforms table metadata into semantic definitions; `validate/` enforces YAML schema and SQL rules; `protos/` stores generated message classes. Iteration flows and partner adapters remain in `journeys/` and `partner/`. Please keep new code co-located with these buckets so imports and caching decorators stay predictable.

## Environment & Build Commands
- Always point Poetry at Python 3.11: run `poetry env use python3.11` (the repo will warn if 3.12 is selected).
- `make setup` (or `poetry install`) resolves deps, including `clickzetta-connector-python` and `clickzetta-zettapark-python`.
- `poetry run streamlit run app.py` launches the ClickZetta semantic generator; `make fmt_lint` applies Black/isort, and `make run_mypy`/`make run_flake8` mirror CI.
- `poetry run pytest -q` executes the unit suite; add `-k clickzetta` for quick connector checks.

## ClickZetta Integration Notes
- Connections read from the same `connections.json` hierarchy used by `mcp-clickzetta-server`. Keep `volume:user://~/semantic_models/` as the default stage and prefer `USE VOLUME` flow—named volumes require extra privileges.
- Metadata queries target ClickZetta’s information schema (`sys.information_schema.tables`) and expect upper-cased identifiers. When adjusting filters, confirm they match https://yunqi.tech/documents/sql-reference semantics.
- Reserved keywords and data types mirror the ClickZetta runtime (`clickzetta/zettapark/_internal/reserved_words.py`). If you add mappings, update `semantic_model_generator/validate/keywords.py` and align `DIMENSION_/MEASURE_/OBJECT_DATATYPES` so unsupported payloads (e.g., VECTOR, VARBINARY, JSON) are skipped gracefully.
- DashScope settings (API key, base URL, model, sampling params) load from the same connection config. When present, the “Use DashScope to enrich descriptions” checkbox will call the model to fill missing descriptions and business-friendly metric aliases; keep the response JSON-only per the prompt schema when changing prompts.
- DashScope prompts and responses are English-only. The enrichment flow and chat assistant both target the `qwen-plus-latest` model; if you adjust prompts, keep the JSON contract (`analysis`/`sql`/`suggestions`) and text output in English.

## Testing & Validation
Add focused tests in `semantic_model_generator/tests`, mirroring the existing `clickzetta_connector_test.py` patterns. Cover both happy paths (table discovery, metadata normalization) and failure modes (permission errors, reserved-word validation). Before shipping ClickZetta SQL helpers, run targeted pytest cases plus at least one manual `poetry run streamlit run app.py` smoke test to verify table selection and YAML output end-to-end.

## Contribution Workflow
Use Conventional Commit prefixes (`feat:`, `fix:`, `docs:`). In PR descriptions, note ClickZetta-specific manual steps (e.g., schema grants required to exercise SHOW TABLES) and include `poetry run pytest` + lint output. When touching configuration or dependency pins, mention downstream impact on `mcp-clickzetta-server` so the teams stay aligned.

## Security & Configuration
Never commit real ClickZetta credentials. Store them in local `.env` files or the shared `connections.json` template referenced in `config/`. Redact customer schema names in screenshots and YAML fixtures. If you need new volumes or external resources, document creation scripts separately and prefer parameterised SQL that avoids hard-coding secrets.
