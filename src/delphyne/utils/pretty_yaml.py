"""
A customized YAML dumper that formats multiline strings in block style.
"""

# pyright: basic

import yaml


class LiteralStr(str):
    pass


def literal_str_representer(dumper, data):
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style="|")


def should_use_block(value):
    return isinstance(value, str) and "\n" in value


def prepare_multiline_strings(obj):
    if isinstance(obj, dict):
        return {k: prepare_multiline_strings(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [prepare_multiline_strings(i) for i in obj]
    elif isinstance(obj, tuple):
        return tuple(prepare_multiline_strings(i) for i in obj)
    elif should_use_block(obj):
        return LiteralStr(obj)
    else:
        return obj


class BlockStyleDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


BlockStyleDumper.add_representer(LiteralStr, literal_str_representer)


def pretty_yaml(str: object, width: int = 100) -> str:
    """
    Pretty-print a value in YAML with multiline strings in block style.
    """
    prepared_data = prepare_multiline_strings(str)
    return yaml.dump(
        prepared_data,
        Dumper=BlockStyleDumper,
        sort_keys=False,
        width=width,
        allow_unicode=True,
        default_flow_style=False,
    )
