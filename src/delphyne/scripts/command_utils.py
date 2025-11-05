from collections.abc import Callable
from typing import Any

from delphyne.utils.yaml import pretty_yaml


def split_command_file(doc: str) -> tuple[str, str | None]:
    """
    Split a command file into its header and outcome sections.

    Returns a tuple of (header, outcome). If there is no outcome
    section, the second element of the tuple will be None.
    """
    lines = doc.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "outcome:":
            header = "\n".join(lines[:idx])
            outcome = "\n".join(lines[idx:])
            return header, outcome
    return doc, None


def command_file_header(doc: str) -> str:
    """
    Remove all lines, following and including the first single line
    equal to `outcome:` (with possibly trailing whitespace).
    """
    header, _ = split_command_file(doc)
    return header


def update_command_file_outcome(doc: str, f: Callable[[Any], Any]) -> str:
    header, outcome = split_command_file(doc)
    if outcome is None:
        return doc
    import yaml

    outcome_data = yaml.safe_load(outcome)["outcome"]
    updated_outcome_data = f(outcome_data)
    outcome = pretty_yaml({"outcome": updated_outcome_data})
    return header.rstrip() + "\n" + outcome
