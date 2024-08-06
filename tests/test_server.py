from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

from delphyne.core.demos import Demonstration
from delphyne.core.strategies import StrategyTree
from delphyne.core.tracing import ExportableTrace
from delphyne.server.evaluate_demo import (
    ExecutionContext,
    evaluate_demo_and_return_tree,
)
from delphyne.server.feedback import DemoFeedback
from delphyne.utils.typing import pydantic_dump, pydantic_load
from delphyne.utils.yaml import dump_yaml


STRATEGY_FILE = "test_strategies"
TESTS_FOLDER = Path(__file__).parent
CONTEXT = ExecutionContext([TESTS_FOLDER], [STRATEGY_FILE])


def test_tracer():
    args = {"allowed": [4, 6, 2, 9], "goal": 11}
    s = CONTEXT.find_and_instantiate_strategy("make_sum", args)
    _tree = StrategyTree.new(s)


def check_object_included(small: object, big: object, path: str = "expect"):

    def error():
        return f"Mismatch at '{path}':\n    Expected: {small}\n    Got: {big}"

    match small:
        case "__any__":
            pass
        case "__empty__":
            assert isinstance(big, (list, tuple)) and not big, error()
        case list() | tuple():
            assert isinstance(big, (list, tuple)), error()
            small = cast(Sequence[Any], small)
            big = cast(Sequence[Any], big)
            assert len(small) <= len(big)
            for i, (small_elt, big_elt) in enumerate(zip(small, big)):
                check_object_included(small_elt, big_elt, f"{path}[{i}]")
        case dict():
            assert isinstance(big, dict), error()
            small = cast(dict[Any, Any], small)
            big = cast(dict[Any, Any], big)
            for k in small:
                assert k in big, error()
                check_object_included(small[k], big[k], f"{path}[{repr(k)}]")
        case _:
            assert small == big, error()


@dataclass
class DemoExpectTest(Demonstration):
    expect: object = None

    def check(self, ctx: ExecutionContext):
        feedback, tree = evaluate_demo_and_return_tree(self, ctx)
        if tree is not None:
            print(dump_yaml(ExportableTrace, tree.tracer.export()))
        print(dump_yaml(DemoFeedback, feedback))
        feedback_serialized = pydantic_dump(
            DemoFeedback, feedback, exclude_defaults=False
        )
        if self.expect is not None:
            check_object_included(self.expect, feedback_serialized)


def load_demo(demo_label: str) -> DemoExpectTest:
    DEMO_FILE = Path(__file__).parent / f"{STRATEGY_FILE}.demo.yaml"
    demos_json = yaml.safe_load(open(DEMO_FILE, "r").read())
    demos = pydantic_load(list[DemoExpectTest], demos_json)
    for demo in demos:
        if demo_label and demo_label == demo.demonstration:
            return demo
    else:
        assert False, f"Not found: {demo_label}"


@pytest.mark.parametrize(
    "demo_label",
    [
        "make_sum_demo",
        "make_sum_stuck",
        "make_sum_selectors",
        "make_sum_at",
        "make_sum_test_parse_error",
        "trivial_strategy",
        "buggy_strategy",
        "strategy_not_found",
        "invalid_arguments",
        "unknown_query",
        "synthetize_fun_demo",
        "invalid_answer",
        "test_iterated",
    ],
)
def test_server(demo_label: str):
    demo = load_demo(demo_label)
    print("\n")
    demo.check(CONTEXT)


if __name__ == "__main__":
    # Entry point for the debugger (see "Debug Server Tests" configuration).
    test_server("test_iterated")
