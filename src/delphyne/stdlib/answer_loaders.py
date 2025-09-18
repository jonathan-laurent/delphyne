"""
A concrete implementation of `AnswerDatabaseLoader`.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import delphyne.core as dp
import delphyne.core.demos as dm
import delphyne.utils.typing as ty
from delphyne.core.traces import ExportableQueryInfo, NodeOriginStr

type _AnswerIterable = Iterable[tuple[dp.SerializedQuery, dp.LocatedAnswer]]


DEMO_FILE_EXT = ".demo.yaml"
COMMAND_FILE_EXT = ".exec.yaml"
COMMAND_RESULT_PATH = ("outcome", "result")
COMMAND_RESULT_TRACE_FIELD = "raw_trace"
COMMAND_RESULT_SUCCESS_NODES_FIELD = "success_nodes"


def standard_answer_loader(workspace_root: Path) -> dp.AnswerDatabaseLoader:
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
        trace, success_nodes = load_trace_and_success_nodes_from_command_file(
            workspace_root / source.command
        )
        node_ids = source.node_ids
        if node_ids is None:
            if success_nodes:
                node_ids = [success_nodes[0]]
            else:
                node_ids = []
        for node_id in node_ids:
            for query, answer_id, answer in relevant_answers(trace, node_id):
                located = dp.LocatedAnswer(
                    answer=answer,
                    source=dp.FromCommandResult(
                        "command_result",
                        source.command,
                        answer_id,
                    ),
                )
                yield (query, located)

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
            `.demo.yaml` will be added if not already present.

    Raises:
        InvalidDemoFile: If the file could not be found or parsed.
    """

    if not path.suffix.endswith(DEMO_FILE_EXT):
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


def load_trace_and_success_nodes_from_command_file(
    path: Path,
) -> tuple[dp.ExportableTrace, Sequence[int]]:
    """
    Load a trace from a command file.
    """

    if not path.suffix:
        path = path.with_suffix(COMMAND_FILE_EXT)
    with open(path, "r") as f:
        content: Any = yaml.safe_load(f)
    for key in COMMAND_RESULT_PATH:
        content = content[key]
    trace_raw = content[COMMAND_RESULT_TRACE_FIELD]
    success_value = content.get(COMMAND_RESULT_SUCCESS_NODES_FIELD, [])
    trace = ty.pydantic_load(dp.ExportableTrace, trace_raw)
    success_nodes = ty.pydantic_load(Sequence[int], success_value)
    return trace, success_nodes


#####
##### Extract answers from traces
#####


def node_and_answer_ids_in_node_origin_string(
    origin: NodeOriginStr,
) -> tuple[set[int], set[int]]:
    """
    Return all node ids and answer ids mentioned in a pretty printed
    node origin reference.

    This is implemented using regexes. Node ids are of the form `%<int>`
    and answer ids are of the form `@<int>`. In addition, `origin` is of
    the form `nested(id, ...)` or `child(id, ...)` and `id` must also be
    added to the sequence of recognized node ids.
    """
    import re

    # Find all %<int> (node ids)
    node_id_matches = re.findall(r"%(\d+)", origin)
    node_ids = set(int(n) for n in node_id_matches)

    # Find all @<int> (answer ids)
    answer_id_matches = re.findall(r"@(\d+)", origin)
    answer_ids = set(int(a) for a in answer_id_matches)

    # Find nested(id, ...) and child(id, ...)
    nested_child_matches = re.findall(r"(?:nested|child)\((\d+)", origin)
    assert len(nested_child_matches) == 1
    node_ids.update(int(n) for n in nested_child_matches)

    return node_ids, answer_ids


def relevant_answers(
    trace: dp.ExportableTrace, node_id: int
) -> Iterable[tuple[dp.SerializedQuery, int, dp.Answer]]:
    """
    Take a trace and a node identifier and return an iterable of all
    answers needed to reach this node in the trace.

    The output can include duplicates.
    """

    answer_info: dict[int, ExportableQueryInfo] = {}
    for query in trace.queries:
        for ans_id in query.answers:
            answer_info[ans_id] = query

    def aux(
        node_id: int,
    ) -> Iterable[tuple[dp.SerializedQuery, int, dp.Answer]]:
        if node_id == 0:
            return
        origin = trace.nodes[node_id]
        nids, aids = node_and_answer_ids_in_node_origin_string(origin)
        for aid in aids:
            info = answer_info[aid]
            assert info.query is not None and info.args is not None, (
                f"Missing query information for answer {aid}."
            )
            query = dp.SerializedQuery.from_json(info.query, info.args)
            yield (query, aid, info.answers[aid])
        for n in nids:
            yield from aux(n)

    yield from aux(node_id)
