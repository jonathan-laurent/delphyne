import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from delphyne.core.queries import Prompt
from delphyne.core.trees import Strategy
from delphyne.server.evaluate_demo import ExecutionContext
from delphyne.stdlib.dsl import strategy
from delphyne.stdlib.nodes import Run, run
from delphyne.stdlib.search_envs import Params, SearchEnv
from delphyne.stdlib.structured import (
    StructuredQuery,
    extract_final_block,
    raw_string,
    raw_yaml,
    string_from_last_block,
)
from delphyne.utils.yaml import dump_yaml, load_yaml


@pytest.mark.parametrize(
    "inp, out",
    [
        (
            """
            The message ends in a code block:

            ```py
            x = 1 + 2
            y = 3
            ```
            """,
            """
            x = 1 + 2
            y = 3
            """,
        ),
        (
            """
            The message ends in a code block:

            ```py
            x = 1 + 2
            y = 3
            ```

            ```
            5 = 6 + 7
            ```
            """,
            """
            5 = 6 + 7
            """,
        ),
    ],
)
def test_extract_final_block(inp: str, out: str | None):
    inp = textwrap.dedent(inp)
    out = textwrap.dedent(out).strip() if out is not None else None
    extracted = extract_final_block(inp)
    assert extracted is None or extracted[-1] == "\n"
    extracted = extracted.strip() if extracted is not None else None
    assert extracted == out


def test_parsers():
    @dataclass
    class Prog:
        content: str

    assert raw_string(str, "hello") == "hello"
    assert string_from_last_block(Prog, "```py\nx = 1\ny = 2\n```") == Prog(
        "x = 1\ny = 2\n"
    )
    assert raw_yaml(dict, "a: 1\nb: 2") == {"a": 1, "b": 2}


@dataclass
class ExamplePrompt(StructuredQuery[Params, list[int]]):
    n: int
    s: str
    d: dict[str, int]


@strategy()
def run_example_prompt(
    n: int, s: str, d: dict[str, int]
) -> Strategy[Run[Params], list[int]]:
    res = yield from run(ExamplePrompt(n, s, d))
    return res


EXPECTED_OUTPUT = """
messages:
  - role: system
    content: |
      Here is an integer: 1.
      Here is a string: hello.
      Here is a dictionary:
      - a: 1
      - b: 2
  - role: user
    content: |
      n: 1
      s: hello
      d: {}
  - role: assistant
    content: '[42]'
  - role: user
    content: |
      n: 1
      s: hello
      d:
        a: 1
        b: 2
options:
  model:
  max_tokens:
"""


def test_structured():
    modules = ["test_prompts", "test_strategies"]
    exe_context = ExecutionContext([Path(__file__).parent], modules)
    demo_paths = [Path(__file__).parent / "examples.demo.yaml"]
    env = SearchEnv(demo_paths, exe_context)
    params = Params(env)
    query = ExamplePrompt(1, "hello", {"a": 1, "b": 2})
    examples = env.collect_examples(query)
    prompt = query.create_prompt(params, examples)
    print(dump_yaml(Prompt, prompt))
    assert prompt == load_yaml(Prompt, EXPECTED_OUTPUT)
