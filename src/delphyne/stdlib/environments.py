"""
Policy environments.

Policies have access to a global environment for fetching prompts, data,
examples, caching LLM requests, and logging information.
"""

from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, override

import jinja2
import yaml

import delphyne.core as dp
import delphyne.core.answer_databases as ad
import delphyne.core.demos as dm
import delphyne.stdlib.answer_loaders as loaders
import delphyne.stdlib.models as md
from delphyne.utils.yaml import dump_yaml_object

####
#### Example Database
####


type _QueryName = str


@dataclass(kw_only=True)
class Example:
    """
    An example, usable for few-shot prompting.

    Attributes:
        query: The corresponding serialized query.
        answer: The answer to the query.
        tags: A sequence of tags associated with the example, which
            policies can use to select appropriate examples.
    """

    query: dp.SerializedQuery
    answer: dp.Answer
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

    def add_query_demonstration(self, demo: dp.QueryDemo):
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
        answer = dm.translate_answer(demo_answer)
        serialized = dp.SerializedQuery.from_json(demo.query, demo.args)
        example = Example(
            query=serialized, answer=answer, tags=demo_answer.tags
        )
        self._examples[demo.query].append(example)

    def add_demonstration(self, demo: dp.Demo):
        """
        Add all examples from a demonstration to the database.
        """
        if isinstance(demo, dp.QueryDemo):
            self.add_query_demonstration(demo)
        else:
            assert isinstance(demo, dp.StrategyDemo)
            for q in demo.queries:
                self.add_query_demonstration(q)

    def examples(self, query: dp.SerializedQuery) -> Iterable[Example]:
        """
        Obtain all potential examples that can be used for few-shot
        prompting with a given query.
        """
        for ex in self._examples[query.name]:
            if self.do_not_match_identical_queries:
                if ex.query == query:
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


def _fail_from_template(msg: str):
    raise jinja2.TemplateError(msg)


class TemplatesManager(dp.AbstractTemplatesManager):
    """
    A class for managing Jinja prompt templates.

    Templates are configured with the `trim_blocks` and `lstrip_blocks`
    options set to `True` (no newlines are inserted after blocks and
    indentation can be used within blocks without affecting the output).
    The `keep_trailing_newline` option is set to `False` so trailing new
    lines at the end of template files are ignored.

    Templates are first searched in the provided prompt folders and then
    in the standard library (`delphyne.stdlib.templates`). For example,
    to show standard formatting instructions, you can include the
    following in your instance prompts:

    ```jinja
    {% include 'stdlib/format.jinja' %}
    ```

    All templates automatically have access to the following global
    objects:

    - A `yaml` filter for converting an object into a YAML string.
    - A `json` filter for converting an object into a JSON string.
    - A `fail` function that takes an error message as an argument and
      raises an exception on Python side.
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
        loader = jinja2.ChoiceLoader(
            [
                jinja2.FileSystemLoader(self.prompt_folders),
                jinja2.PackageLoader("delphyne.stdlib"),
            ]
        )
        self.env = jinja2.Environment(
            loader=loader,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=False,
        )
        self.env.filters["yaml"] = dump_yaml_object
        self.env.filters["json"] = _dump_json_object
        self.env.globals["fail"] = _fail_from_template  # type: ignore

    @override
    def prompt(
        self,
        *,
        query_name: str,
        prompt_kind: Literal["system", "instance"] | str,
        template_args: dict[str, Any],
        default_template: str | None = None,
    ) -> str:
        suffix = "." + prompt_kind
        template_name = f"{query_name}{suffix}{JINJA_EXTENSION}"
        try:
            template = self.env.get_template(template_name)
        except jinja2.TemplateNotFound:
            if default_template is not None:
                template = self.env.from_string(default_template)
            else:
                raise dp.TemplateFileMissing(template_name)
        try:
            assert "data" not in template_args
            template_args |= {"data": self.data}
            return template.render(template_args)
        except jinja2.TemplateNotFound as e:
            raise dp.TemplateError(template_name, e)
        except jinja2.UndefinedError as e:
            raise dp.TemplateError(template_name, e)
        except jinja2.TemplateSyntaxError as e:
            raise dp.TemplateError(template_name, e)


def _dump_json_object(
    obj: object,
    *,
    exclude_defaults: bool = False,
    exclude_none: bool = False,
    exclude_fields: Iterable[str] = (),
    indent: int | str | None = None,
):
    import json

    import pydantic

    Adapter = pydantic.TypeAdapter(type(obj))
    py = Adapter.dump_python(
        obj,
        exclude_defaults=exclude_defaults,
        exclude_none=exclude_none,
        warnings="error",
    )
    if isinstance(py, dict):
        py = cast(dict[Any, Any], py)
        for f in exclude_fields:
            del py[f]
    return json.dumps(py, indent=indent)


####
#### Hindsight Feedback Data
####


@dataclass(frozen=True)
class HindsightFeedback:
    """
    Feedback about what the answer to a query *should have been*.
    """

    query: str
    args: dict[str, object]
    answer: dp.Answer


type HindsightFeedbackDict = dict[int, HindsightFeedback]
"""
A mapping from node IDs to the attached hindsight feedback.
"""


####
#### Policy Environment
####


class PolicyEnv:
    """
    The global environment accessible to policies.

    It can be used for:

    - Fetching prompts, data, and examples.
    - Caching LLM requests.
    - Tracing nodes, query, answers, and logging information.

    Attributes:
        cache: The (optional) request cache.
        templates: The prompt templates manager.
        tracer: The tracer, which can also be used for logging.
        examples: The example database.
        log_long_computations: see constructor.
    """

    def __init__(
        self,
        *,
        prompt_dirs: Sequence[Path] = (),
        demonstration_files: Sequence[Path] = (),
        data_dirs: Sequence[Path] = (),
        cache: md.LLMCache | None = None,
        override_answers: dp.AnswerDatabase | None = None,
        log_level: dp.LogLevel = "info",
        log_long_computations: tuple[dp.LogLevel, float] | None = None,
        do_not_match_identical_queries: bool = False,
    ):
        """
        Args:
            prompt_dirs: A sequence of directories where Jinja prompt
                templates can be found.
            demonstration_files: A sequence of paths to demonstration
                files (with or without extension `.demo.yaml`), to
                create an example database from.
            data_dirs: A sequence of directories where data files can be
                found.
            cache: A request cache, or `None` to disable caching.
            override_answers: If provided, a database of answers that
                must be used to override LLM calls whenever possible.
                Individual prompting policies such as `few_shot` are
                responsible for consulting this global database using
                the `overriden_answer` method.
            log_level: The minimum log level to record. Messages with a
                lower level will be ignored.
            log_long_computations: if set, log computations taking more
                than the given number of seconds at the given severity
                level. This settings can be locally overriden by
                `elim_compute`.
            do_not_match_identical_queries: See `ExampleDatabase`.
        """
        self.templates = TemplatesManager(prompt_dirs, data_dirs)
        self.examples = ExampleDatabase(do_not_match_identical_queries)
        self.tracer = dp.Tracer(log_level=log_level)
        self.log_long_computations = log_long_computations
        self.cache = cache
        self.override_answers = override_answers
        for path in demonstration_files:
            for demo in loaders.load_demo_file(path):
                self.examples.add_demonstration(demo)
        self._hindsight_feedback: HindsightFeedbackDict = {}

    def add_hindsight_feedback(
        self, node_id: int, feedback: HindsightFeedback
    ) -> None:
        with self.tracer.lock:
            self._hindsight_feedback[node_id] = feedback

    def get_hindsight_feedback(self) -> HindsightFeedbackDict:
        with self.tracer.lock:
            return self._hindsight_feedback.copy()

    def overriden_answer(
        self, query: dp.AbstractQuery[Any]
    ) -> dp.Answer | None:
        """
        Attempt to fetch an answer from the override database and return
        it if it exists, while logging the event.
        """

        if self.override_answers is None:
            return None
        serialized_query = dp.SerializedQuery.make(query)
        ret = self.override_answers.fetch(serialized_query)
        if ret is not None:
            meta = {
                "source": ad.pp_located_answer_source(ret.source),
                "query_name": query.query_name(),
                "query_args": query.serialize_args(),
                "answer": ret.answer,
            }
            self.info("llm_override", meta)
            return ret.answer

    def log(
        self,
        level: dp.LogLevel,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
    ) -> None:
        """
        Log a message.

        Arguments:
            level: The severity level of the message.
            message: The message to log.
            metadata: Additional metadata to log, as a dictionary of JSON
                values.
            loc: Tree or attached query that the message is about, if
                relevant.
        """

        match loc:
            case None:
                location = None
            case dp.Tree():
                location = dp.Location(loc.ref, None)
            case dp.AttachedQuery(_, ref):
                location = dp.Location(ref[0], ref[1])
        self.tracer.log(level, message, metadata, location)

    def trace(
        self,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
    ) -> None:
        """
        Log a message with "trace" severity level.

        See `log` method.
        """
        self.log("trace", message, metadata, loc=loc)

    def debug(
        self,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
    ) -> None:
        """
        Log a message with "debug" severity level.

        See `log` method.
        """
        self.log("debug", message, metadata, loc=loc)

    def info(
        self,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
    ) -> None:
        """
        Log a message with "info" severity level.

        See `log` method.
        """
        self.log("info", message, metadata, loc=loc)

    def warn(
        self,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
    ) -> None:
        """
        Log a message with "warn" severity level.

        See `log` method.
        """
        self.log("warn", message, metadata, loc=loc)

    def error(
        self,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
    ) -> None:
        """
        Log a message with "error" severity level.

        See `log` method.
        """
        self.log("error", message, metadata, loc=loc)
