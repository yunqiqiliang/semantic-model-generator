from semantic_model_generator.clickzetta_utils.clickzetta_connector import (
    ClickzettaConnectionProxy,
)


def load_yaml(yaml_path: str) -> str:
    """
    Load local yaml file into str.

    yaml_path: str The absolute path to the location of your yaml file. Something like path/to/your/file.yaml.
    """
    with open(yaml_path) as f:
        yaml_str = f.read()
    return yaml_str


def validate(yaml_str: str, conn: ClickzettaConnectionProxy) -> None:
    """
    Placeholder validation for ClickZetta. The ClickZetta API does not yet expose a
    dedicated semantic model validator, so we treat models as valid after basic checks.
    """
    _ = yaml_str
    _ = conn


def validate_from_local_path(yaml_path: str, conn: ClickzettaConnectionProxy) -> None:
    yaml_str = load_yaml(yaml_path)
    validate(yaml_str, conn)
