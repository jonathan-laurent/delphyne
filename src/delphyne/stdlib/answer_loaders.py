"""
A concrete implementation of `AnswerDatabaseLoader`.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

import delphyne.core as dp
import delphyne.core.demos as dm
import delphyne.utils.typing as ty

type _AnswerIterable = Iterable[tuple[dp.SerializedQuery, dp.LocatedAnswer]]


DEMO_FILE_EXT = ".demo.yaml"
COMMAND_FILE_EXT = ".exec.yaml"


def standard_answer_loader(
    workspace_root: Path,
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
        assert False

    def loader(source: dp.AnswerSource) -> _AnswerIterable:
        match source:
            case dp.CommandResultAnswerSource():
                return command_result_loader(source)
            case dp.DemoAnswerSource():
                return demo_loader(source)

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
