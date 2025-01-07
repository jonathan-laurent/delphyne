"""
Policy environments.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jinja2
import yaml

from delphyne.core import refs, traces
from delphyne.core.demos import Demonstration
from delphyne.core.refs import Answer
from delphyne.utils.typing import pydantic_load

type QueryArgs = dict[str, Any]


####
#### Example Database
####


@dataclass
class ExampleDatabase:
    """
    A simple example database. Examples are stored as JSON strings.

    TODO: add provenance info for better error messages.
    """

    _examples: dict[str, list[tuple[QueryArgs, Answer]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_demonstration(self, demo: Demonstration):
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


####
#### Jinja Prompts
####


PROMPT_DIR = "prompts"
SYSTEM_MESSAGE_SUFFIX = ""
INSTANCE_MESSAGE_SUFFIX = ".instance"


class TemplatesManager:
    def __init__(self, strategy_dirs: Sequence[Path]):
        prompt_folders = [dir / PROMPT_DIR for dir in strategy_dirs]
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(prompt_folders),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def prompt(
        self,
        type: Literal["system", "instance"],
        query_name: str,
        template_args: dict[str, Any],
    ) -> str:
        """
        Raises `TemplateNotFound`.
        """
        if type == "system":
            suffix = SYSTEM_MESSAGE_SUFFIX
        elif type == "instance":
            suffix = INSTANCE_MESSAGE_SUFFIX
        template_name = f"{query_name}{suffix}.jinja"
        try:
            template = self.env.get_template(template_name)
        except jinja2.TemplateNotFound as e:
            raise TemplateNotFound(e)
        return template.render(template_args)


@dataclass
class TemplateNotFound(Exception):
    exn: Exception


####
#### Tracer
####


@dataclass(frozen=True)
class LogMessage:
    message: str
    metadata: dict[str, Any] | None = None
    location: traces.Location | None = None


class Tracer:
    def __init__(self):
        self.trace = traces.Trace()
        self.messages: list[LogMessage] = []

    def trace_space(self, ref: refs.GlobalSpacePath) -> None:
        self.trace.add_location(traces.Location(ref[0], ref[1]))

    def trace_node(self, node: refs.GlobalNodePath) -> None:
        self.trace.add_location(traces.Location(node, None))

    def log(
        self,
        message: str,
        metadata: dict[str, Any] | None = None,
        location: traces.Location | None = None,
    ):
        if location is not None:
            self.trace.add_location(location)
        self.messages.append(LogMessage(message, metadata, location))


####
#### Policy Environment
####


class PolicyEnv:
    def __init__(
        self,
        strategy_dirs: Sequence[Path],
        demonstration_files: Sequence[Path],
    ):
        """
        An environment accessible to a policy, containing prompt and
        example databases in particular.

        The `strategy_dirs` argument is used to deduce a directory for
        prompt templates.
        """
        self.templates = TemplatesManager(strategy_dirs)
        self.examples = ExampleDatabase()
        self.tracer = Tracer()
        for path in demonstration_files:
            with path.open() as f:
                demos = pydantic_load(list[Demonstration], yaml.safe_load(f))
                for demo in demos:
                    self.examples.add_demonstration(demo)
