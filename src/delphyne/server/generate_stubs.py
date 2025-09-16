"""
Automatically generate Typescript translations for demos.py and
feedback.py.

An LLM is called to make the translation so make sure to put your OpenAI
API key in `OPENAI_API_KEY`. Also, always check the diff before
committing the changes.

Usage: python -m delphyne.server.generate_stubs <demos|feedback>
"""

import sys
import textwrap
from pathlib import Path

from openai import OpenAI


def text(s: str) -> str:
    return textwrap.dedent(s).replace("\n", " ").strip()


def convert_python_to_typescript(
    python_code: str,
    instructions: list[str],
) -> str:
    client = OpenAI()

    system_prompt = text(
        """
        I am going to give you the content of a Python file. You must
        identify all type and dataclass definitions in this file and
        convert them to Typescript. Please answer with the Typescript
        code and nothing else. Do not put your answer inside a
        triple-backquoted block. Do not include comments in the result.
        Use Typescript interfaces and not classes. Export every
        definition. Convert `Path` to `string`. Convert `object` to
        `unknown`. Convert `None` to `null`. Use two spaces for
        indentation and one blank line between each definition. Add a
        semicolon at the end of each type definition.` Represent python
        dictionaries using `Record`. For example, the type `dict[str,
        int]` should be converted to `Record<string, int>`. Fields for
        which a default value is provided must be labelled as optional
        (using `?`).

        Examples:
        - `foo: str | None` becomes `foo: string | null;` (important:
           not an optional field since no default value is provided)
        - `bar: str | None = None` becomes `bar?: string | null;`
        - `quux: dict[str, int] = {}` becomes `quux?: Record<string, int>;`
        """
    )
    system_prompt = "\n\n".join([system_prompt] + instructions)

    response = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": python_code},
        ],
        max_tokens=1500,
        temperature=0.3,
    )

    answer = response.choices[0].message.content
    assert answer is not None
    return answer


def convert_stub(file: Path, instructions: list[str]):
    with open(file, "r") as f:
        python_code = f.read()
    print(convert_python_to_typescript(python_code, instructions))


if __name__ == "__main__":
    match sys.argv:
        case [_, "demos"]:
            demo_file = Path(__file__).parent.parent / "core" / "demos.py"
            instructions = [
                text(
                    """
                    Only process the file up to the DemoFile type
                    definition. Do NOT include all definitions from
                    `NodeLabel` to the end of the file.
                    """
                ),
                text(
                    """
                    In each dataclass definition, add a field `__loc` of
                    type `vscode.Range`. Also, for each field `foo`, add
                    a field `__loc__foo`. In addition, for each field
                    `bar` that is an array, add a field
                    `__loc_items__bar` of type `vscode.Range[]` right
                    after `__loc__bar`. All
                    these special location fields should be at the end
                    of the translated interface definition.
                    """
                ),
                text(
                    """
                    Start with the following import: `import * as vscode
                    from 'vscode';
                    """
                ),
            ]
            convert_stub(demo_file, instructions)
        case [_, "feedback"]:
            file = Path(__file__).parent.parent / "analysis" / "feedback.py"
            convert_stub(file, [])
        case args:
            assert False, f"Unknown arguments: {args}"
