import os
import yaml
import json
import pandas as pd
from typing import Dict, Any

def load_config(config_path: str = "src/configs/default_config.yaml") -> Dict[str, Any]:
    """Loads the YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def ensure_dir(path: str):
    """Ensures a directory exists."""
    os.makedirs(path, exist_ok=True)

def save_json(data: Any, filepath: str):
    """Saves data to a JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

def load_json(filepath: str) -> Any:
    """Loads data from a JSON file."""
    with open(filepath, "r") as f:
        return json.load(f)

def save_metrics(metrics: Dict[str, Any], filepath: str):
    """Saves metrics to a CSV file (appends if exists) or JSON."""
    # specific logic if needed, for now alias to save_json
    save_json(metrics, filepath)
