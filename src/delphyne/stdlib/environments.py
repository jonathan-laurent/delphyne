"""
Policy environments.

Policies have access to a global environment for fetching prompts, data,
examples, caching LLM requests, and logging information.
"""

import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, override

import jinja2
import numpy as np
import yaml
from numpy.typing import NDArray

import delphyne.core as dp
import delphyne.core.answer_databases as ad
import delphyne.core.demos as dm
import delphyne.stdlib.answer_loaders as loaders
import delphyne.stdlib.embeddings as em
import delphyne.stdlib.models as md
from delphyne.analysis import ObjectLoader
from delphyne.utils.yaml import dump_yaml_object

####
#### Data Manager
####


class DataManager:
    """
    Utility class for loading and accessing external data.

    Attributes:
        data: A dictionary containing all loaded data files. Each file
            corresponds to a key in the dictionary (stripped of the
            extension).
    """

    def __init__(self, data_dirs: Sequence[Path]):
        """
        Find all files with extension `*.data.yaml` in the `data_dirs`,
        parse them and save everything in a big dict. If two files have
        the same name, raise an error.
        """
        self.data = _load_data(data_dirs)


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


####
#### Example Database
####


type _QueryName = str


type _EmbeddingModelMame = str


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


class ExampleDatabase:
    """
    A simple example database.
    """

    # TODO: add provenance info for better error messages.

    def __init__(
        self,
        *,
        embeddings_cache_file: Path | None = None,
        templates_manager: dp.AbstractTemplatesManager | None = None,
        object_loader: ObjectLoader | None = None,
        do_not_match_identical_queries: bool = False,
    ):
        """
        Arguments:
            embeddings_cache_file: Global cache file that stores
                common embeddings (e.g. embeddings of examples).
            templates_manager: A templates manager, necessary when using
                embeddings.
            object_loader: An object loader, necessary when using
                embeddings.
            do_not_match_identical_queries: If set to `True`, the
                `examples` method won't return examples that match
                identical queries (i.e., with the exact same arguments).
                This is useful in the context of writing demonstrations,
                where one may want to see how an LLM would answer a
                query, even when a ground-truth answer is provided
                already.
        """
        self._embeddings_cache_file = embeddings_cache_file
        self._templates_manager = templates_manager
        self._object_loader = object_loader
        self._do_not_match_identical_queries = do_not_match_identical_queries
        self._examples: dict[_QueryName, list[Example]] = defaultdict(list)
        self._embeddings: dict[
            tuple[_QueryName, _EmbeddingModelMame], NDArray[np.float64]
        ] = {}

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

    def examples(self, query: dp.AbstractQuery[Any]) -> Iterable[Example]:
        """
        Obtain all potential examples that can be used for few-shot
        prompting with a given query.
        """
        serialized = dp.SerializedQuery.make(query)
        for ex in self._examples[serialized.name]:
            if self._do_not_match_identical_queries:
                if ex.query == serialized:
                    continue
            yield ex

    def all_examples(self, query_name: str) -> Sequence[Example]:
        """
        Obtain examples by their indices for a given query name.
        """
        return self._examples[query_name]

    ### Useful accessors

    @property
    def templates_manager(self) -> dp.AbstractTemplatesManager:
        if self._templates_manager is None:
            raise ValueError(
                "ExampleDatabase.templates_manager was not provided."
            )
        return self._templates_manager

    @property
    def object_loader(self) -> ObjectLoader:
        if self._object_loader is None:
            raise ValueError("ExampleDatabase.object_loader was not provided.")
        return self._object_loader

    @property
    def embeddings_cache_file(self) -> Path:
        if self._embeddings_cache_file is None:
            raise ValueError(
                "ExampleDatabase.embeddings_cache_file was not provided."
            )
        return self._embeddings_cache_file

    ### Loading embeddings

    def embeddings_for_query_type(
        self,
        name: _QueryName,
        model: _EmbeddingModelMame,
    ) -> NDArray[np.float64]:
        """
        Obtain the embeddings for all examples of a given query type.

        If the embeddings are not loaded yet, they are loaded from
        cache or computed on the fly.
        """
        key = (name, model)
        if key not in self._embeddings:
            self.load_embeddings_for_query_type(name, model)
        return self._embeddings[key]

    def load_embeddings_for_query_type(
        self,
        name: _QueryName,
        model: _EmbeddingModelMame,
    ) -> None:
        """
        Get emebdings for all examples of a given query type.

        This method takes a global file lock so as to avoid concurrent
        accesses to the embeddings cache file.
        """
        import filelock

        lock_file = _embeddings_cache_lockfile(self.embeddings_cache_file)
        examples = self._examples[name]
        with filelock.FileLock(lock_file):
            res = self._load_embeddings(model, examples)
        self._embeddings[(name, model)] = res

    def embedding_text(self, query: dp.AbstractQuery[Any]) -> str:
        return _query_embedding_text(self.templates_manager, query)

    def _embedding_text_from_serialized(
        self, query: dp.SerializedQuery
    ) -> str:
        loaded = self.object_loader.load_query(query.name, query.args_dict)
        return self.embedding_text(loaded)

    def _load_embeddings(
        self,
        model_name: _EmbeddingModelMame,
        examples: Sequence[Example],
    ) -> NDArray[np.float64]:
        model = em.standard_openai_embedding_model(model_name)
        to_embed = [
            self._embedding_text_from_serialized(ex.query) for ex in examples
        ]
        cache_file = self.embeddings_cache_file
        with md.load_request_cache(cache_file, mode="read_write") as cache:
            res = model.embed(to_embed, cache)
        embeddings = np.array([r.embedding for r in res], dtype=np.float64)
        # We ignore spending for the global cache.
        return embeddings


def _embeddings_cache_lockfile(cache_file: Path) -> Path:
    return cache_file.with_suffix(cache_file.suffix + ".lock")


def _query_embedding_text(
    templates_manager: dp.AbstractTemplatesManager,
    query: dp.AbstractQuery[Any],
) -> str:
    return query.generate_prompt(
        kind=em.EMBEDDING_PROMPT_NAME,
        mode=None,
        params={},
        extra_args=None,
        env=templates_manager,
    )


####
#### Jinja Prompts
####


JINJA_EXTENSION = ".jinja"


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
    - A `data` dictionary containing all loaded data files.
    """

    def __init__(self, prompt_dirs: Sequence[Path], data_manager: DataManager):
        """
        Args:
            prompt_dirs: A sequence of directories where Jinja prompt
                templates can be found.
            data_manager: A sequence of directories where data files can be
                found.
        """
        self.prompt_folders = prompt_dirs
        self.data_manager = data_manager
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
            template_args |= {"data": self.data_manager.data}
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
        data_manager: The data manager.
        templates: The prompt templates manager.
        tracer: The tracer, which can also be used for logging.
        examples: The example database.
        log_long_computations: See constructor.
        random: A random number generator.
    """

    def __init__(
        self,
        *,
        prompt_dirs: Sequence[Path] = (),
        demonstration_files: Sequence[Path] = (),
        data_dirs: Sequence[Path] = (),
        cache: md.LLMCache | None = None,
        embeddings_cache_file: Path | None = None,
        object_loader: ObjectLoader | None = None,
        override_answers: dp.AnswerDatabase | None = None,
        log_level: dp.LogLevel = "info",
        log_long_computations: tuple[dp.LogLevel, float] | None = None,
        do_not_match_identical_queries: bool = False,
        random_seed: int = 0,
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
            embeddings_cache_file: Global cache file that stores
                common embeddings (e.g. embeddings of examples).
            override_answers: If provided, a database of answers that
                must be used to override LLM calls whenever possible.
                Individual prompting policies such as `few_shot` are
                responsible for consulting this global database using
                the `overriden_answer` method.
            object_loader: An object loader. This is useful for
                computing query embeddings, which requires parsing
                serialized queries.
            log_level: The minimum log level to record. Messages with a
                lower level will be ignored.
            log_long_computations: if set, log computations taking more
                than the given number of seconds at the given severity
                level. This settings can be locally overriden by
                `elim_compute`.
            do_not_match_identical_queries: See `ExampleDatabase`.
            random_seed: The seed with which to initialize the random
                number generator.
        """
        self.data_manager = DataManager(data_dirs)
        self.templates = TemplatesManager(prompt_dirs, self.data_manager)
        self.examples = ExampleDatabase(
            embeddings_cache_file=embeddings_cache_file,
            do_not_match_identical_queries=do_not_match_identical_queries,
        )
        self._object_loader = object_loader
        self.tracer = dp.Tracer(log_level=log_level)
        self.log_long_computations = log_long_computations
        self.cache = cache
        self.override_answers = override_answers
        self.random = random.Random(random_seed)
        for path in demonstration_files:
            for demo in loaders.load_demo_file(path):
                self.examples.add_demonstration(demo)

    @property
    def object_loader(self) -> ObjectLoader:
        """
        The object loader associated with this environment.

        A runtime error is raised if no object loader was provided.
        """
        if self._object_loader is None:
            raise ValueError("PolicyEnv.object_loader was not provided.")
        return self._object_loader

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
        location = loc.ref if loc is not None else None
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
