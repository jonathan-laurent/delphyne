"""
A concrete implementation of `AnswerDatabaseLoader`.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import delphyne.analysis as an
import delphyne.core as dp
import delphyne.core.demos as dm
import delphyne.stdlib.feedback_processing as fp
import delphyne.utils.typing as ty

type _AnswerIterable = Iterable[tuple[dp.SerializedQuery, dp.LocatedAnswer]]


DEMO_FILE_EXT = ".demo.yaml"
COMMAND_FILE_EXT = ".exec.yaml"
COMMAND_ARGS_PATH = ("args",)
COMMAND_STRATEGY_NAME_FIELD = "strategy"
COMMAND_STRATEGY_ARGS_FIELD = "args"
COMMAND_RESULT_PATH = ("outcome", "result")
COMMAND_RESULT_TRACE_FIELD = "raw_trace"
COMMAND_RESULT_SUCCESS_NODES_FIELD = "success_nodes"


def standard_answer_loader(
    workspace_root: Path, object_loader: an.ObjectLoader
) -> dp.AnswerDatabaseLoader:
    """
    Standard answer loader.
    """

    def load_from_query_demonstration(
        demo: dp.QueryDemo, source: dp.LocatedAnswerSource
    ) -> _AnswerIterable:
        if not demo.answers:
            return
        serialized = dp.SerializedQuery.from_json(demo.query, demo.args)
        answer = dp.LocatedAnswer(
            answer=dm.translate_answer(demo.answers[0]),
            source=source,
        )
        yield (serialized, answer)

    def demo_loader(source: dp.DemoAnswerSource) -> _AnswerIterable:
        file, demo_name = source.file_and_demo_name
        demos = load_demo_file(workspace_root / file)
        demo = demo_with_name(demos, demo_name)
        if isinstance(demo, dm.QueryDemo):
            yield from load_from_query_demonstration(
                demo,
                source=dp.FromStandaloneQueryDemo(
                    "standalone_query_demo", file, demo_name
                ),
            )
        else:
            for i, query in enumerate(demo.queries):
                yield from load_from_query_demonstration(
                    query,
                    source=dp.FromStrategyDemo(
                        "strategy_demo", file, demo_name, i, 0
                    ),
                )

    def command_result_loader(
        source: dp.CommandResultAnswerSource,
    ) -> _AnswerIterable:
        command_file = workspace_root / source.command
        trace_data = load_trace_data_from_command_file(command_file)
        node_ids = source.node_ids
        if node_ids is None:
            if trace_data.success_nodes:
                node_ids = [trace_data.success_nodes[0]]
            else:
                node_ids = []
        resolver = trace_data.resolver(object_loader)
        raw_examples = fp.extract_examples(
            resolver,
            roots=[dp.irefs.NodeId(i) for i in node_ids],
            backprop_handler_tags=source.backprop_with,
        )
        for ex in raw_examples:
            serialized = dp.SerializedQuery.make(ex.query)
            answer = dp.LocatedAnswer(
                answer=ex.answer,
                source=dp.FromCommandResult(
                    "command_result",
                    source.command,
                    answer_id=ex.answer_id.id,
                    modified=ex.modified,
                ),
            )
            yield (serialized, answer)

    def unfiltered_loader(source: dp.AnswerSource) -> _AnswerIterable:
        match source:
            case dp.CommandResultAnswerSource():
                return command_result_loader(source)
            case dp.DemoAnswerSource():
                return demo_loader(source)

    def loader(source: dp.AnswerSource) -> _AnswerIterable:
        filter = source.queries
        if filter is None:
            yield from unfiltered_loader(source)
        else:
            for query, answer in unfiltered_loader(source):
                if query.name in filter:
                    yield (query, answer)

    return loader


@dataclass
class InvalidDemoFile(Exception):
    """
    Exception raised when a demonstration file could not be parsed.
    """

    file: Path
    exn: Exception


def load_demo_file(path: Path) -> Sequence[dm.Demo]:
    """
    Load a demonstration file from the given path.

    Arguments:
        path: The path to the demonstration file. The suffix
            `.demo.yaml` will be added if no extension is given.

    Raises:
        InvalidDemoFile: If the file could not be found or parsed.
    """

    if not path.suffix:
        path = path.with_suffix(DEMO_FILE_EXT)
    try:
        with path.open() as f:
            content = yaml.safe_load(f)
            demos = ty.pydantic_load(Sequence[dp.Demo], content)
            return demos
    except Exception as e:
        raise InvalidDemoFile(path, e)


def demo_with_name(demos: Sequence[dm.Demo], name: str) -> dm.Demo:
    """
    Find a demonstration with the given name in a sequence of
    demonstrations.

    Arguments:
        demos: A sequence of demonstrations.
        name: The name of the demonstration to find.

    Raises:
        ValueError: If no demonstration with the given name is found or
            multiple demonstrations with the given name are found.
    """

    cands = [demo for demo in demos if demo.demonstration == name]
    if len(cands) > 1:
        raise ValueError(f"Multiple demonstrations named '{name}' found")
    if len(cands) == 1:
        return cands[0]
    raise ValueError(f"No demonstration named '{name}' found")


@dataclass
class TraceData:
    strategy: str
    args: dict[str, Any]
    trace: dp.ExportableTrace
    success_nodes: Sequence[int]

    def resolver(self, object_loader: an.ObjectLoader) -> an.IRefResolver:
        trace = dp.Trace.load(self.trace)
        strategy = object_loader.load_strategy_instance(
            self.strategy, self.args
        )
        return an.IRefResolver(trace, root=dp.reify(strategy))


def load_trace_data_from_command_file(path: Path) -> TraceData:
    """
    Load trace-related data from a command file.
    """

    if not path.suffix:
        path = path.with_suffix(COMMAND_FILE_EXT)
    with open(path, "r") as f:
        content: Any = yaml.safe_load(f)
    cmd_args = content
    for key in COMMAND_ARGS_PATH:
        cmd_args = cmd_args[key]
    strategy = cmd_args[COMMAND_STRATEGY_NAME_FIELD]
    args = cmd_args.get(COMMAND_STRATEGY_ARGS_FIELD, {})
    for key in COMMAND_RESULT_PATH:
        content = content[key]
    trace_raw = content[COMMAND_RESULT_TRACE_FIELD]
    success_value = content.get(COMMAND_RESULT_SUCCESS_NODES_FIELD, [])
    trace = ty.pydantic_load(dp.ExportableTrace, trace_raw)
    success_nodes = ty.pydantic_load(Sequence[int], success_value)
    return TraceData(strategy, args, trace, success_nodes)
