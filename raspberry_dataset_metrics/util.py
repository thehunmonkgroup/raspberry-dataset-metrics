from pathlib import Path

def get_config_base_name(config_path: Path | str) -> str:
    return Path(config_path).stem
