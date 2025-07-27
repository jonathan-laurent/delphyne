"""
Loading configuration files
"""

from pathlib import Path
from typing import Any

import yaml

import delphyne.stdlib as std
import delphyne.utils.typing as ty

WORKSPACE_FILE = "delphyne.yaml"


def load_config(workspace_dir: Path) -> std.CommandExecutionContext:
    config_path = workspace_dir / WORKSPACE_FILE
    if config_path.exists():
        with open(config_path, "r") as f:
            data: Any = yaml.safe_load(f) or {}
    else:
        data = {}
    cex = ty.pydantic_load(std.CommandExecutionContext, data)
    return cex.with_root(workspace_dir)
