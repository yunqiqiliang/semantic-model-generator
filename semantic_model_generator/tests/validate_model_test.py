from types import SimpleNamespace

from semantic_model_generator.validate_model import validate


class DummyConnection:
    def __init__(self):
        self.session = SimpleNamespace()


def test_validate_returns_none_for_clickzetta_proxy() -> None:
    conn = DummyConnection()
    yaml_str = "tables: []"
    assert validate(yaml_str, conn) is None
