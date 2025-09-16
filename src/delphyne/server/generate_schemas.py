"""
Generate schemas to be used by the editor extension.

Usage: python -m delphyne.server.generate_schemas demo_file
"""

import json
import sys

import pydantic

from delphyne.core.demos import DemoFile
from delphyne.stdlib import CommandExecutionContext


def demo_file_schema():
    schema = pydantic.TypeAdapter(DemoFile).json_schema()
    schema["title"] = "Delphyne Demo File"
    return schema


def config_file_schema():
    schema = pydantic.TypeAdapter(CommandExecutionContext).json_schema()
    schema["title"] = "Delphyne Configuration File"
    assert "workspace_root" in schema["properties"]
    del schema["properties"]["workspace_root"]
    return schema


if __name__ == "__main__":
    match sys.argv:
        case [_, "demo_file"]:
            print(json.dumps(demo_file_schema(), indent=2))
        case [_, "config_file"]:
            print(json.dumps(config_file_schema(), indent=2))
        case _:
            pass
