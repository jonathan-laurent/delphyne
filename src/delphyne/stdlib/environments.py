"""
Policy environments.

Policies have access to a global environment for fetching prompts, data,
examples, caching LLM requests, and logging information.
"""

import random
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
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
from delphyne.utils.yaml import dump_yaml_object, pretty_yaml

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


type _EmbeddingModelName = str


type _EmbeddingsBucket = tuple[_QueryName, _EmbeddingModelName]


@dataclass(kw_only=True)
class Example:
    """
    An example, usable for few-shot prompting.

    Attributes:
        query: The corresponding query.
        answer: The answer to the query.
        tags: A sequence of tags associated with the example, which
            policies can use to select appropriate examples.
    """

    query: dp.AbstractQuery[Any]
    answer: dp.Answer
    tags: Sequence[str]
    meta: dict[str, Any]


class ExampleDatabase:
    """
    A simple example database.
    """

    # TODO: add provenance info for better error messages.

    def __init__(
        self,
        *,
        object_loader: ObjectLoader,
        global_embeddings_cache_file: Path | None = None,
        templates_manager: dp.AbstractTemplatesManager | None = None,
    ):
        """
        Arguments:
            object_loader: An object loader for loading query objects.
            global_embeddings_cache_file: Global cache file that stores
                common embeddings (e.g. embeddings of examples).
            templates_manager: A templates manager, necessary when using
                embeddings.
        """
        self._embeddings_cache_file = global_embeddings_cache_file
        self.object_loader = object_loader
        self._templates_manager = templates_manager
        self._examples: dict[_QueryName, list[Example]] = defaultdict(list)

        # For both `_query_embeddings` and `_example_embeddings`, we
        # store `None` if the embeddings were computed but there were
        # zero examples. This is the equivalent of a numpy array with
        # zero lines.

        # Embeddings for queries, classified by type
        self._query_embeddings: dict[
            _EmbeddingsBucket, NDArray[np.float32] | None
        ] = {}
        # Embeddings for full examples, classified by type
        self._example_embeddings: dict[
            _EmbeddingsBucket, NDArray[np.float32] | None
        ] = {}
        # Similarity matrix
        self._example_similarity_matrix: dict[
            _EmbeddingsBucket, NDArray[np.float32] | None
        ] = {}

    def add_demonstration(self, demo: dp.Demo):
        """
        Add all examples from a demonstration to the database.
        """
        if isinstance(demo, dp.QueryDemo):
            self._add_query_demonstration(demo)
        else:
            assert isinstance(demo, dp.StrategyDemo)
            for q in demo.queries:
                self._add_query_demonstration(q)
        # Embeddings are cleared since new examples were added.
        self._query_embeddings.clear()

    def _add_query_demonstration(self, demo: dp.QueryDemo):
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
        query = self.object_loader.load_query(demo.query, demo.args)
        example = Example(
            query=query,
            answer=answer,
            tags=demo_answer.tags,
            meta=demo_answer.meta or {},
        )
        self._examples[demo.query].append(example)

    def examples_for(self, query_name: str) -> Sequence[Example]:
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
    def global_embeddings_cache_file(self) -> Path:
        if self._embeddings_cache_file is None:
            raise ValueError(
                "ExampleDatabase.global_embeddings_cache_file was not provided."
            )
        return self._embeddings_cache_file

    ### Loading embeddings

    def query_embedding_text(self, query: dp.AbstractQuery[Any]) -> str:
        return _query_embedding_text(self.templates_manager, query)

    def example_embedding_text(self, example: Example) -> str:
        return _example_embedding_text(self.templates_manager, example)

    def fetch_query_embeddings(
        self,
        name: _QueryName,
        model: _EmbeddingModelName,
    ) -> NDArray[np.float32] | None:
        """
        Obtain the query embeddings for all examples of a given type.

        If the embeddings are not loaded yet, they are loaded from
        cache or computed on the fly.
        """
        key = (name, model)
        if key not in self._query_embeddings:
            embs = self._load_embeddings(
                model,
                self.examples_for(name),
                lambda e: self.query_embedding_text(e.query),
            )
            self._query_embeddings[key] = embs
        return self._query_embeddings[key]

    def fetch_example_embeddings(
        self,
        name: _QueryName,
        model: _EmbeddingModelName,
    ) -> NDArray[np.float32] | None:
        """
        Obtain the embeddings of all examples of a given type.

        If the embeddings are not loaded yet, they are loaded from
        cache or computed on the fly.
        """
        key = (name, model)
        if key not in self._example_embeddings:
            embs = self._load_embeddings(
                model,
                self.examples_for(name),
                self.example_embedding_text,
            )
            self._example_embeddings[key] = embs
        return self._example_embeddings[key]

    def fetch_example_similarity_matrix(
        self,
        name: _QueryName,
        model: _EmbeddingModelName,
    ) -> NDArray[np.float32] | None:
        """
        Obtain the similarity matrix of all examples of a given type.

        If the similarity matrix is not loaded yet, it is computed on
        the fly from the example embeddings.
        """
        key = (name, model)
        if key not in self._example_similarity_matrix:
            embs = self.fetch_example_embeddings(name, model)
            if embs is None:
                sim_matrix = None
            else:
                # Compute cosine similarity matrix
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                normalized_embs = embs / np.clip(
                    norms, a_min=1e-10, a_max=None
                )
                sim_matrix = normalized_embs @ normalized_embs.T
            self._example_similarity_matrix[key] = sim_matrix
        return self._example_similarity_matrix[key]

    def _load_embeddings(
        self,
        model_name: _EmbeddingModelName,
        examples: Sequence[Example],
        embed_fun: Callable[[Example], str],
    ) -> NDArray[np.float32] | None:
        """
        Get embeddings for all examples of a given query type.

        This method takes a global file lock so as to avoid concurrent
        accesses to the embeddings cache file.
        """

        import filelock

        # Note: on Unix systems, the lockfile may not be automatically
        # deleted. See https://stackoverflow.com/questions/58098634/

        model = em.standard_openai_embedding_model(model_name)
        to_embed = [embed_fun(e) for e in examples]
        cache_file = self.global_embeddings_cache_file
        lock_file = _embeddings_cache_lockfile(
            self.global_embeddings_cache_file
        )
        with filelock.FileLock(lock_file):
            with em.load_embeddings_cache(cache_file, "read_write") as cache:
                res = model.embed(to_embed, cache)
        if not res:
            return None
        embeddings = np.array([r.embedding for r in res], dtype=np.float32)
        # We ignore spending for the global cache.
        return embeddings


def _embeddings_cache_lockfile(cache_file: Path) -> Path:
    return cache_file.with_suffix(cache_file.suffix + ".lock")


def _query_embedding_text(
    templates_manager: dp.AbstractTemplatesManager,
    query: dp.AbstractQuery[Any],
) -> str:
    return query.generate_prompt(
        kind=em.QUERY_EMBEDDING_PROMPT_NAME,
        mode=None,
        params={},
        extra_args=None,
        env=templates_manager,
    )


def _example_embedding_text(
    templates_manager: dp.AbstractTemplatesManager,
    example: Example,
) -> str:
    answer = example.answer
    if isinstance(answer.content, str):
        rendered = answer.content
    else:
        rendered = pretty_yaml(answer.content.structured)
    return example.query.generate_prompt(
        kind=em.EXAMPLE_EMBEDDING_PROMPT_NAME,
        mode=None,
        params={},
        extra_args={"answer": rendered},
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
        object_loader: ObjectLoader,
        prompt_dirs: Sequence[Path] = (),
        demonstration_files: Sequence[Path] = (),
        data_dirs: Sequence[Path] = (),
        cache: md.LLMCache | None = None,
        embeddings_cache: em.EmbeddingsCache | None = None,
        global_embeddings_cache_file: Path | None = None,
        override_answers: dp.AnswerDatabase | None = None,
        log_level: dp.LogLevel = "info",
        log_long_computations: tuple[dp.LogLevel, float] | None = None,
        random_seed: int = 0,
    ):
        """
        Args:
            object_loader: An object loader. This is useful in
                particular for loading query objects from their
                serialized representation.
            prompt_dirs: A sequence of directories where Jinja prompt
                templates can be found.
            demonstration_files: A sequence of paths to demonstration
                files (with or without extension `.demo.yaml`), to
                create an example database from.
            data_dirs: A sequence of directories where data files can be
                found.
            cache: A request cache, or `None` to disable caching.
            embeddings_cache: An embeddings cache, or `None` to disable
                embeddings caching.
            global_embeddings_cache_file: Global cache file that stores
                common embeddings (e.g. embeddings of examples).
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
            random_seed: The seed with which to initialize the random
                number generator.
        """
        self.data_manager = DataManager(data_dirs)
        self.templates = TemplatesManager(prompt_dirs, self.data_manager)
        self.examples = ExampleDatabase(
            global_embeddings_cache_file=global_embeddings_cache_file,
            templates_manager=self.templates,
            object_loader=object_loader,
        )
        self.object_loader = object_loader
        self.tracer = dp.Tracer(log_level=log_level)
        self.log_long_computations = log_long_computations
        self.cache = cache
        self.embeddings_cache = embeddings_cache
        self.override_answers = override_answers
        self.random = random.Random(random_seed)
        for path in demonstration_files:
            for demo in loaders.load_demo_file(path):
                self.examples.add_demonstration(demo)

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
        related: Sequence[dp.LogMessageId | None] = (),
    ) -> dp.LogMessageId | None:
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
        return self.tracer.log(
            level, message, metadata, location=location, related=related
        )

    def trace(
        self,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
        related: Sequence[dp.LogMessageId | None] = (),
    ) -> dp.LogMessageId | None:
        """
        Log a message with "trace" severity level.

        See `log` method.
        """
        return self.log("trace", message, metadata, loc=loc, related=related)

    def debug(
        self,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
        related: Sequence[dp.LogMessageId | None] = (),
    ) -> dp.LogMessageId | None:
        """
        Log a message with "debug" severity level.

        See `log` method.
        """
        return self.log("debug", message, metadata, loc=loc, related=related)

    def info(
        self,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
        related: Sequence[dp.LogMessageId | None] = (),
    ) -> dp.LogMessageId | None:
        """
        Log a message with "info" severity level.

        See `log` method.
        """
        return self.log("info", message, metadata, loc=loc, related=related)

    def warn(
        self,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
        related: Sequence[dp.LogMessageId | None] = (),
    ) -> dp.LogMessageId | None:
        """
        Log a message with "warn" severity level.

        See `log` method.
        """
        return self.log("warn", message, metadata, loc=loc, related=related)

    def error(
        self,
        message: str,
        metadata: object | None = None,
        *,
        loc: dp.Tree[Any, Any, Any] | dp.AttachedQuery[Any] | None = None,
        related: Sequence[dp.LogMessageId | None] = (),
    ) -> dp.LogMessageId | None:
        """
        Log a message with "error" severity level.

        See `log` method.
        """
        return self.log("error", message, metadata, loc=loc, related=related)
