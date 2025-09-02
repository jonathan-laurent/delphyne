"""
Generating README.md

The goal of this script is to generate the README.md file from the
docs/index.md file. This is done in two passes:

1. Make the substitutions specified in the `readme_diff.yaml` file. All
   strings mentioned in this file must be trimmed. A warning is issues
   if a string to be replaced is not found.
2. Convert all remaining relative markdown links into absolute links to
   the documentation. Specifically, every markdown link of the form
   `[...](./xxx.md#id)` must be translated into
   `[...]({SITE_BASE}/xxx/#id)`.
3. Remove header ids of the form `{#...}`.

The script emits the resulting README.md file on stdout.
"""

import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

SITE_BASE = "https://jonathan-laurent.github.io/delphyne/latest"


def load_readme_diff(
    readme_diff_path: Union[str, Path],
) -> List[Dict[str, Any]]:
    """Load the substitutions from readme_diff.yaml"""
    with open(readme_diff_path, "r") as f:
        return yaml.safe_load(f)


def apply_substitutions(
    content: str, substitutions: List[Dict[str, Any]]
) -> str:
    """Apply the substitutions specified in readme_diff.yaml"""
    for sub in substitutions:
        replace_text = sub["replace"].strip()
        by_text = sub["by"].strip()

        if replace_text in content:
            content = content.replace(replace_text, by_text)
        else:
            print(
                f"Warning: String to be replaced not found:\n{replace_text}",
                file=sys.stderr,
            )

    return content


def convert_relative_links(content: str, site_base: str) -> str:
    """Convert relative markdown links to absolute links"""
    # Pattern to match markdown links of the form [text](./path.md) or [text](./path.md#id)
    pattern = r"\[([^\]]+)\]\(\./([^)#]+)\.md(#[^)]+)?\)"

    def replace_link(match: re.Match[str]) -> str:
        text = match.group(1)
        path = match.group(2)
        fragment = match.group(3) if match.group(3) else ""
        return f"[{text}]({site_base}/{path}/{fragment})"

    return re.sub(pattern, replace_link, content)


def remove_header_ids(content: str) -> str:
    """Remove header ids of the form {#...}"""
    pattern = r" \{#[^}]*\}"
    return re.sub(pattern, "", content)


def main() -> None:
    # Determine script directory and related paths
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    docs_index_path = repo_root / "docs" / "index.md"
    readme_diff_path = script_dir / "readme_diff.yaml"

    # Read the source content
    with open(docs_index_path, "r") as f:
        content = f.read()

    # Load substitutions
    substitutions = load_readme_diff(readme_diff_path)

    # Apply substitutions
    content = apply_substitutions(content, substitutions)

    # Convert relative links to absolute links
    content = convert_relative_links(content, SITE_BASE)

    # Remove header ids
    content = remove_header_ids(content)

    # Output the result
    print(content)


if __name__ == "__main__":
    main()
