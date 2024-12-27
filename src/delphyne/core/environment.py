"""
Policy environments.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from delphyne.core import demos
from delphyne.core.refs import Answer


type QueryArgs = dict[str, Any]


@dataclass
class ExampleDatabase:
    """
    A simple example database. Examples are stored as JSON strings. We
    do not need to create explicit.

    TODO: add provenance info for better error messages.
    """

    _examples: dict[str, list[tuple[QueryArgs, Answer]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_demonstration(self, demo: demos.Demonstration):
        for q in demo.queries:
            if not q.answers:
                continue
            if (ex := q.answers[0].example) is not None and not ex:
                # If the user explicitly asked not to
                # include the example. TODO: What if the user
                # asked to include several answers?
                continue
            answer = q.answers[0].answer
            mode = q.answers[0].mode
            self._examples[q.query].append((q.args, Answer(mode, answer)))

    def examples(self, query_name: str) -> Sequence[tuple[QueryArgs, Answer]]:
        return self._examples[query_name]
