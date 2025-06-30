"""
Policy environments.
"""

import json
import threading
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jinja2
import yaml

from delphyne.core import pprint, refs, traces
from delphyne.core.demos import Demo, QueryDemo, StrategyDemo, translate_answer
from delphyne.core.refs import Answer
from delphyne.utils.typing import pydantic_load
from delphyne.utils.yaml import dump_yaml_object

type _QueryName = str

type QueryArgs = dict[str, Any]


def _equal_query_args(args1: QueryArgs, args2: QueryArgs) -> bool:
    # Comparing the dictionaries directly would not work because the
    # same object where a tuple is used instead of a list would be
    # considered different.
    return json.dumps(args1) == json.dumps(args2)


@dataclass
class Example:
    args: QueryArgs
    answer: Answer
    tags: Sequence[str]


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

    # Maps each query name to a list of
    _examples: dict[_QueryName, list[Example]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_query_demonstration(self, demo: QueryDemo):
        if not demo.answers:
            return
        if (ex := demo.answers[0].example) is not None and not ex:
            # If the user explicitly asked not to include the example.
            # TODO: What if the user asked to include several answers?
            # Right now, we only allow the first one to be added.
            return
        demo_answer = demo.answers[0]
        answer = translate_answer(demo_answer)
        example = Example(demo.args, answer, demo_answer.tags)
        self._examples[demo.query].append(example)

    def add_demonstration(self, demo: Demo):
        if isinstance(demo, QueryDemo):
            self.add_query_demonstration(demo)
        else:
            assert isinstance(demo, StrategyDemo)
            for q in demo.queries:
                self.add_query_demonstration(q)

    def examples(
        self, query_name: str, query_args: QueryArgs
    ) -> Iterable[Example]:
        for ex in self._examples[query_name]:
            if self.do_not_match_identical_queries:
                if _equal_query_args(ex.args, query_args):
                    continue
            yield ex


####
#### Jinja Prompts
####


JINJA_EXTENSION = ".jinja"


def _load_data(data_dirs: Sequence[Path]) -> dict[str, Any]:
    # Find all files with extension `*.data.yaml` in the data_dirs,
    # parse them and save everything in a big dict. If two files have
    # the same name, raise an error.
    result: dict[str, Any] = {}
    seen_filenames: set[str] = set()

    for data_dir in data_dirs:
        if not data_dir.exists():
            continue

        for yaml_file in data_dir.glob("*.data.yaml"):
            filename = yaml_file.name

            # Check for duplicate filenames
            if filename in seen_filenames:
                raise ValueError(f"Duplicate data file found: {filename}")

            seen_filenames.add(filename)

            # Parse the YAML file and add its contents to the result
            try:
                with yaml_file.open() as f:
                    data = yaml.safe_load(f)
                    if data is not None:
                        # Use the filename (without extension) as the key
                        key = yaml_file.stem.replace(".data", "")
                        result[key] = data
            except Exception as e:
                raise ValueError(f"Error parsing YAML file {yaml_file}: {e}")

    return result


class TemplatesManager:
    def __init__(self, prompt_dirs: Sequence[Path], data_dirs: Sequence[Path]):
        self.prompt_folders = prompt_dirs
        self.data = _load_data(data_dirs)
        self.env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(self.prompt_folders),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.env.filters["yaml"] = dump_yaml_object  # type: ignore

    def prompt(
        self,
        kind: Literal["system", "instance"] | str,
        query_name: str,
        template_args: dict[str, Any],
        default_template: str | None = None,
    ) -> str:
        suffix = "." + kind
        template_name = f"{query_name}{suffix}{JINJA_EXTENSION}"
        prompt_file_exists = any(
            (d / template_name).exists() for d in self.prompt_folders
        )
        if not prompt_file_exists:
            if default_template is not None:
                template = self.env.from_string(default_template)
            else:
                raise TemplateFileMissing(template_name)
        else:
            template = self.env.get_template(template_name)
        try:
            assert "data" not in template_args
            template_args |= {"data": self.data}
            return template.render(template_args)
        except jinja2.TemplateNotFound as e:
            raise TemplateError(template_name, e)
        except jinja2.UndefinedError as e:
            raise TemplateError(template_name, e)
        except jinja2.TemplateSyntaxError as e:
            raise TemplateError(template_name, e)


@dataclass
class TemplateError(Exception):
    name: str
    exn: Exception


@dataclass
class TemplateFileMissing(Exception):
    """
    We want to make a distinction with the `TemplateNotFound` Jinja
    exception, which can also be raised when `include` statements fail
    within templates. In comparison, this exception means that the main
    template file does not exist.
    """

    file: str


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

        # Different threads may be logging information or appending to
        # the trace in parallel.
        self.lock = threading.RLock()

    def trace_query(self, ref: refs.GlobalSpacePath) -> None:
        """
        Ensure that a query at a given reference is present in the trace.
        """
        with self.lock:
            self.trace.convert_query_origin(ref)

    def trace_space(self, ref: refs.GlobalSpacePath) -> None:
        """
        TODO: currently unused.
        """
        with self.lock:
            self.trace.convert_location(traces.Location(ref[0], ref[1]))

    def trace_node(self, node: refs.GlobalNodePath) -> None:
        """
        Ensure that a node at a given reference is present in the trace.
        """
        with self.lock:
            self.trace.convert_location(traces.Location(node, None))

    def trace_answer(
        self, space: refs.GlobalSpacePath, answer: refs.Answer
    ) -> None:
        with self.lock:
            self.trace.convert_answer_ref((space, answer))

    def log(
        self,
        message: str,
        metadata: dict[str, Any] | None = None,
        location: traces.Location | None = None,
    ):
        with self.lock:
            short_location = None
            if location is not None:
                short_location = self.trace.convert_location(location)
            self.messages.append(LogMessage(message, metadata, short_location))

    def export_log(self) -> Iterable[ExportableLogMessage]:
        with self.lock:
            for m in self.messages:
                node = None
                space = None
                if (loc := m.location) is not None:
                    node = loc.node.id
                    if loc.space is not None:
                        space = pprint.space_ref(loc.space)
                yield ExportableLogMessage(m.message, node, space, m.metadata)

    def export_trace(self) -> traces.ExportableTrace:
        with self.lock:
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
        prompt_dirs: Sequence[Path],
        demonstration_files: Sequence[Path],
        data_dirs: Sequence[Path],
        do_not_match_identical_queries: bool = False,
    ):
        """
        An environment accessible to a policy, containing prompt and
        example databases in particular.
        """
        self.templates = TemplatesManager(prompt_dirs, data_dirs)
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
