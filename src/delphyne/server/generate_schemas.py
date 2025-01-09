"""
Generate schemas to be used by the editor extension.

Usage: python -m delphyne.server.generate_schemas demo_file
"""

import json
import sys

import pydantic

from delphyne.core.demos import DemoFile


def demo_file_schema():
    schema = pydantic.TypeAdapter(DemoFile).json_schema()
    schema["title"] = "Delphyne Demo File"
    return schema


if __name__ == "__main__":
    match sys.argv:
        case [_, "demo_file"]:
            print(json.dumps(demo_file_schema(), indent=2))
        case _:
            pass
