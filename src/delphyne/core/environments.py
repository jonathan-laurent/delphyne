"""
Policy environments.

Policies have access to a global environment for fetching prompts, data,
examples, caching LLM requests, and logging information.
"""

import json
import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import jinja2
import yaml

from delphyne.core import pprint, refs, traces
from delphyne.core.demos import Demo, QueryDemo, StrategyDemo, translate_answer
from delphyne.core.refs import Answer
from delphyne.utils.caching import CacheSpec
from delphyne.utils.typing import pydantic_load
from delphyne.utils.yaml import dump_yaml_object

####
#### Example Database
####


DEMO_FILE_EXT = ".demo.yaml"


type _QueryName = str


type QuerySerializedArgs = dict[str, Any]
"""
Serialized query arguments, as a dictionary mapping attributed to JSON
values (assemblies of integers, strings, dictionnaries, list, tuples...).
"""


def _equal_query_args(
    args1: QuerySerializedArgs, args2: QuerySerializedArgs
) -> bool:
    # Comparing the dictionaries directly would not work because the
    # same object where a tuple is used instead of a list would be
    # considered different.
    return json.dumps(args1) == json.dumps(args2)


@dataclass
class Example:
    """
    An example, usable for few-shot prompting.

    Attributes:
        args: The serialized query arguments.
        answer: The answer to the query.
        tags: A sequence of tags associated with the example, which
            policies can use to select appropriate examples.
    """

    args: QuerySerializedArgs
    answer: Answer
    tags: Sequence[str]


@dataclass
class ExampleDatabase:
    """
    A simple example database.

    Attributes:
        do_not_match_identical_queries: If set to `True`, the `examples`
            method won't return examples that match identical queries
            (i.e., with the exact same arguments). This is useful in the
            context of writing demonstrations, where one may want to see
            how an LLM would answer a query, even when a ground-truth
            answer is provided already.

    TODO: add provenance info for better error messages.
    """

    do_not_match_identical_queries: bool = False

    # Maps each query name to a list of
    _examples: dict[_QueryName, list[Example]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_query_demonstration(self, demo: QueryDemo):
        """
        Add all examples from a standalone query demonstration to the
        database.
        """
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
        """
        Add all exmples from a demonstration to the database.
        """
        if isinstance(demo, QueryDemo):
            self.add_query_demonstration(demo)
        else:
            assert isinstance(demo, StrategyDemo)
            for q in demo.queries:
                self.add_query_demonstration(q)

    def examples(
        self, query_name: str, query_args: QuerySerializedArgs
    ) -> Iterable[Example]:
        """
        Obtain all potential examples that can be used for few-shot
        prompting with a given query.
        """
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
    """
    Load all data entries, which are then made accessible in prompts.

    Find all files with extension `*.data.yaml` in the data_dirs,
    parse them and save everything in a big dict. If two files have
    the same name, raise an error.
    """
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
    """
    A class for managing Jinja prompt templates.
    """

    def __init__(self, prompt_dirs: Sequence[Path], data_dirs: Sequence[Path]):
        """
        Args:
            prompt_dirs: A sequence of directories where Jinja prompt
                templates can be found.
            data_dirs: A sequence of directories where data files can be
                found.
        """
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
        *,
        query_name: str,
        prompt_kind: Literal["system", "instance"] | str,
        template_args: dict[str, Any],
        default_template: str | None = None,
    ) -> str:
        """
        Render a prompt message using a template.

        Args:
            query_name: The name of the query for which the prompt is
                built. Used to determine the template file name, namely
                "{query_name}.{prompt_kind}.jinja".
            kind: The kind of prompt (e.g. "system" or "instance") that
                is being rendered, used to determine the name of the
                template file to use.
            template_args: A dictionary of arguments to pass to the
                template. It must not contain key "data", which is
                reserved for the data loaded from the data directories.
            default_template: If provided, this template will be used if
                no template file is found for the given query name and
                kind instead of raising an error.

        Raises:
            TemplateFileMissing: template file not found.
            TemplateError: error raised while rendering the template.
        """

        suffix = "." + prompt_kind
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
    """
    Wrapper for template-related exceptions.
    """

    name: str
    exn: Exception


@dataclass
class TemplateFileMissing(Exception):
    """
    Exception raised when a template file is missing.

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
        *,
        prompt_dirs: Sequence[Path],
        demonstration_files: Sequence[Path],
        data_dirs: Sequence[Path],
        cache: CacheSpec | None = None,
        make_cache: Callable[[CacheSpec], object] | None = None,
        do_not_match_identical_queries: bool = False,
    ):
        """
        An environment accessible to a policy, containing prompt and
        example databases in particular.
        """
        self.templates = TemplatesManager(prompt_dirs, data_dirs)
        self.examples = ExampleDatabase(do_not_match_identical_queries)
        self.tracer = Tracer()
        self.requests_cache = None
        if cache is not None:
            assert make_cache is not None
            self.requests_cache = make_cache(cache)
        for path in demonstration_files:
            if not path.suffix.endswith(DEMO_FILE_EXT):
                path = path.with_suffix(DEMO_FILE_EXT)
            try:
                with path.open() as f:
                    content = yaml.safe_load(f)
                    demos = pydantic_load(list[Demo], content)
                    for demo in demos:
                        self.examples.add_demonstration(demo)
            except Exception as e:
                raise InvalidDemoFile(path, e)
