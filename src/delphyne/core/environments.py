"""
Policy environments.
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jinja2
import yaml

from delphyne.core import pprint, refs, traces
from delphyne.core.demos import Demo, QueryDemo, StrategyDemo
from delphyne.core.refs import Answer
from delphyne.utils.typing import pydantic_load

type QueryArgs = dict[str, Any]


def _equal_query_args(args1: QueryArgs, args2: QueryArgs) -> bool:
    # Comparing the dictionaries directly would not work because the
    # same object where a tuple is used instead of a list would be
    # considered different.
    return json.dumps(args1) == json.dumps(args2)


####
#### Example Database
####


@dataclass
class ExampleDatabase:
    """
    A simple example database. Examples are stored as JSON strings.

    The `do_not_match_identical_queries` parameter can be set to `True`
    in a context where demonstrations are being developed. Indeed, it is
    not interesting to see what the LLM would answer if the solution is
    in the context.

    TODO: add provenance info for better error messages.
    """

    do_not_match_identical_queries: bool = False
    _examples: dict[str, list[tuple[QueryArgs, Answer]]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_query_demonstration(self, demo: QueryDemo):
        if not demo.answers:
            return
        if (ex := demo.answers[0].example) is not None and not ex:
            # If the user explicitly asked not to
            # include the example. TODO: What if the user
            # asked to include several answers?
            return
        answer = demo.answers[0].answer
        mode = demo.answers[0].mode
        self._examples[demo.query].append((demo.args, Answer(mode, answer)))

    def add_demonstration(self, demo: Demo):
        if isinstance(demo, QueryDemo):
            self.add_query_demonstration(demo)
        else:
            assert isinstance(demo, StrategyDemo)
            for q in demo.queries:
                self.add_query_demonstration(q)

    def examples(
        self, query_name: str, query_args: QueryArgs
    ) -> Iterable[tuple[QueryArgs, Answer]]:
        for ex in self._examples[query_name]:
            if self.do_not_match_identical_queries:
                if _equal_query_args(ex[0], query_args):
                    continue
            yield ex


####
#### Jinja Prompts
####


PROMPT_DIR = "prompts"
JINJA_EXTENSIONS = [".jinja", ".md.jinja"]


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
        kind: Literal["system", "instance"] | str,
        query_name: str,
        template_args: dict[str, Any],
    ) -> str:
        """
        Raises `TemplateNotFound`.
        """
        suffix = "." + kind
        for ext in JINJA_EXTENSIONS:
            template_name = f"{query_name}{suffix}{ext}"
            try:
                template = self.env.get_template(template_name)
                return template.render(template_args)
            except jinja2.TemplateNotFound:
                pass
            except jinja2.UndefinedError as e:
                raise TemplateError(template_name, e)
            except jinja2.TemplateSyntaxError as e:
                raise TemplateError(template_name, e)
        raise TemplateNotFound(f"{query_name}{suffix}.*.jinja")


@dataclass
class TemplateNotFound(Exception):
    exn: str


@dataclass
class TemplateError(Exception):
    name: str
    exn: Exception


####
#### Tracer
####


@dataclass(frozen=True)
class LogMessage:
    message: str
    metadata: dict[str, Any] | None = None
    location: traces.ShortLocation | None = None


@dataclass(frozen=True)
class ExportableLogMessage:
    message: str
    node: int | None
    space: str | None
    metadata: dict[str, Any] | None = None


class Tracer:
    def __init__(self):
        self.trace = traces.Trace()
        self.messages: list[LogMessage] = []

    def trace_space(self, ref: refs.GlobalSpacePath) -> None:
        self.trace.convert_location(traces.Location(ref[0], ref[1]))

    def trace_node(self, node: refs.GlobalNodePath) -> None:
        self.trace.convert_location(traces.Location(node, None))

    def log(
        self,
        message: str,
        metadata: dict[str, Any] | None = None,
        location: traces.Location | None = None,
    ):
        short_location = None
        if location is not None:
            short_location = self.trace.convert_location(location)
        self.messages.append(LogMessage(message, metadata, short_location))

    def export_log(self) -> Iterable[ExportableLogMessage]:
        for m in self.messages:
            node = None
            space = None
            if (loc := m.location) is not None:
                node = loc.node.id
                if loc.space is not None:
                    space = pprint.space_ref(loc.space)
            yield ExportableLogMessage(m.message, node, space, m.metadata)

    def export_trace(self) -> traces.ExportableTrace:
        return self.trace.export()


####
#### Policy Environment
####


@dataclass
class InvalidDemoFile(Exception):
    file: Path
    exn: Exception


class PolicyEnv:
    def __init__(
        self,
        strategy_dirs: Sequence[Path],
        demonstration_files: Sequence[Path],
        do_not_match_identical_queries: bool = False,
    ):
        """
        An environment accessible to a policy, containing prompt and
        example databases in particular.

        The `strategy_dirs` argument is used to deduce a directory for
        prompt templates.
        """
        self.templates = TemplatesManager(strategy_dirs)
        self.examples = ExampleDatabase(do_not_match_identical_queries)
        self.tracer = Tracer()
        for path in demonstration_files:
            try:
                with path.open() as f:
                    content = yaml.safe_load(f)
                    demos = pydantic_load(list[Demo], content)
                    for demo in demos:
                        self.examples.add_demonstration(demo)
            except Exception as e:
                raise InvalidDemoFile(path, e)
