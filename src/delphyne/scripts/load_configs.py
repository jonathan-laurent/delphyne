"""
Loading configuration files
"""

from pathlib import Path
from typing import Any

import yaml

import delphyne.stdlib as std
import delphyne.utils.typing as ty

WORKSPACE_FILE = "delphyne.yaml"


def load_config(
    workspace_dir: Path, local_config_from: Path | None
) -> std.CommandExecutionContext:
    config_path = workspace_dir / WORKSPACE_FILE
    if config_path.exists():
        with open(config_path, "r") as f:
            data: Any = yaml.safe_load(f) or {}
    else:
        data = {}
    # If local_config_from is provided, merge local config
    if local_config_from is not None:
        with open(local_config_from, "r") as f:
            document = f.read()
        local_block = extract_config_block(document)
        if local_block is not None:
            local_data: Any = yaml.safe_load(local_block) or {}
            assert isinstance(local_data, dict)
            data = data | local_data
    cex = ty.pydantic_load(std.CommandExecutionContext, data)
    return cex.with_root(workspace_dir)


#####
##### Finding workspace directory
#####


def find_workspace_dir(starting_dir: Path) -> Path | None:
    """
    Find the workspace directory by looking for the delphyne.yaml file.
    """
    current_dir = starting_dir.resolve()
    while current_dir != current_dir.parent:
        if (current_dir / WORKSPACE_FILE).exists():
            return current_dir
        current_dir = current_dir.parent
    return None


#####
##### Extracting local configuration blocks
#####


def extract_config_block(document: str) -> str | None:
    lines = document.split("\n")
    config_start_index = -1
    config_end_index = -1

    # Find the start of the config block
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # If we encounter a non-comment, non-blank line before finding
        # @config, return None
        if line_stripped and not line_stripped.startswith("#"):
            return None
        if line_stripped == "# @config":
            config_start_index = i
            break
    if config_start_index == -1:
        return None

    # Find the end of the config block and validate content
    for i in range(config_start_index + 1, len(lines)):
        line_stripped = lines[i].strip()
        if line_stripped == "# @end":
            config_end_index = i
            break
        if line_stripped and not line_stripped.startswith("#"):
            return None
    if config_end_index == -1:
        return None

    # Extract the content between @config and @end
    config_lines: list[str] = []
    for i in range(config_start_index + 1, config_end_index):
        line = lines[i]
        if line.startswith("# "):
            config_lines.append(line[2:])
        elif line.strip() == "#":
            config_lines.append("")
        else:
            config_lines.append(line)
    return "\n".join(config_lines)
