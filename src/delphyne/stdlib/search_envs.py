"""
Utilities for manipulating search parameters.
"""

import json
from collections import defaultdict
from collections.abc import Awaitable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import yaml

from delphyne.core.demos import Demonstration
from delphyne.core.queries import AnyQuery, Prompt
from delphyne.server.evaluate_demo import ExecutionContext
from delphyne.stdlib import openai_util
from delphyne.stdlib.generators import Budget, PromptHook
from delphyne.stdlib.jinja_prompts import JinjaPromptManager
from delphyne.utils.typing import pydantic_load


@dataclass
class BasicExampleDatabase:
    demo_paths: Sequence[Path]
    exe_context: ExecutionContext

    def __post_init__(self):
        # Map from (query, json serialized args) to answer
        self.examples: dict[str, dict[str, tuple[AnyQuery, str]]] = (
            defaultdict(dict)
        )
        for path in self.demo_paths:
            with path.open() as f:
                demos = pydantic_load(list[Demonstration], yaml.safe_load(f))
                for d in demos:
                    for q in d.queries:
                        if not q.answers:
                            continue
                        if (ex := q.answers[0].example) is not None and not ex:
                            # If the user explicitly asked not to
                            # include the example. TODO: What if the user
                            # asked to include several answers?
                            continue
                        answer = q.answers[0].answer
                        query = self.exe_context.load_query(q.query, q.args)
                        serialized = json.dumps(q.args)
                        self.examples[q.query][serialized] = (query, answer)
        pass

    def collect_examples(
        self, query: AnyQuery
    ) -> Sequence[tuple[AnyQuery, str]]:
        serialized = json.dumps(query.serialize_args())
        cands = self.examples[query.name()]
        selected = [ex for k, ex in cands.items() if k != serialized]
        return selected


@dataclass
class SearchEnv:
    demo_paths: Sequence[Path]
    exe_context: ExecutionContext

    def __post_init__(self):
        self.jinja = JinjaPromptManager(self.exe_context.strategy_dirs)
        self.examples_database = BasicExampleDatabase(
            self.demo_paths, self.exe_context
        )
        self.prompt_hook: PromptHook = lambda _p, _aid: None

    def set_prompt_hook(self, hook: PromptHook) -> None:
        self.prompt_hook = hook

    def execute_prompt(
        self, query: AnyQuery, prompt: Prompt, n: int
    ) -> Awaitable[tuple[Sequence[str | None], Budget]]:
        return openai_util.execute_prompt(prompt, n)

    def collect_examples(
        self, query: AnyQuery
    ) -> Sequence[tuple[AnyQuery, str]]:
        return self.examples_database.collect_examples(query)

    def estimate_cost(self, prompt: Prompt) -> Budget:
        return openai_util.estimate_cost(prompt)


class HasSearchEnv(Protocol):
    @property
    def env(self) -> SearchEnv: ...


@dataclass
class Params:
    env: SearchEnv
