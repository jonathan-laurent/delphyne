"""
Testing the demonstration interpreter with _expect tests_.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
import yaml

import delphyne as dp
from delphyne import analysis
from delphyne.analysis import feedback as fb
from delphyne.utils import typing as ty
from delphyne.utils.yaml import dump_yaml

STRATEGY_FILE = "example_strategies"
TESTS_FOLDER = Path(__file__).parent
CONTEXT = analysis.DemoExecutionContext([TESTS_FOLDER], [STRATEGY_FILE])
LOADER = analysis.ObjectLoader(CONTEXT)


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
class DemoExpectTest(dp.Demonstration):
    expect: object = None

    def check(self, loader: analysis.ObjectLoader):
        feedback, trace = analysis.evaluate_demo_and_return_trace(self, loader)
        if trace is not None:
            print(dump_yaml(dp.ExportableTrace, trace.export()))
        print(dump_yaml(fb.DemoFeedback, feedback))
        feedback_serialized = ty.pydantic_dump(
            fb.DemoFeedback, feedback, exclude_defaults=False
        )
        if self.expect is not None:
            check_object_included(self.expect, feedback_serialized)


def load_demo(demo_label: str) -> DemoExpectTest:
    DEMO_FILE = Path(__file__).parent / f"{STRATEGY_FILE}.demo.yaml"
    demos_json = yaml.safe_load(open(DEMO_FILE, "r").read())
    demos = ty.pydantic_load(list[DemoExpectTest], demos_json)
    for demo in demos:
        if demo_label and demo_label == demo.demonstration:
            return demo
    else:
        assert False, f"Not found: {demo_label}"


@pytest.mark.parametrize(
    "demo_label",
    [
        "make_sum_demo",
        "make_sum_selectors",
        "make_sum_at",
        "make_sum_stuck",
        "make_sum_test_parse_error",
        "trivial_strategy",
        "buggy_strategy",
        "strategy_not_found",
        "invalid_arguments",
        "unknown_query",
        "invalid_answer",
        "synthetize_fun_demo",
        "test_iterated",
    ],
)
def test_server(demo_label: str):
    demo = load_demo(demo_label)
    print("\n")
    demo.check(LOADER)


if __name__ == "__main__":
    # Entry point for the debugger (see "Debug Server Tests" configuration).
    test_server("test_iterated")
