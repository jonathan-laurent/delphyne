import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pytest

import why3_utils as why3


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("x >= y", ["x", "y"]),
        ("a + b * c", ["a", "b", "c"]),
        ("foo == bar && baz != qux", ["foo", "bar", "baz", "qux"]),
        ("_x34-y_", ["_x34", "y_"]),
    ],
)
def test_get_variable_names(expr: str, expected: list[str]) -> None:
    f = why3._get_variable_names  # pyright: ignore[reportPrivateUsage]
    result = f(expr)
    print(result)
    assert sorted(result) == sorted(expected)


@pytest.mark.parametrize(
    "assumptions,conclusion,expected",
    [
        (["x >= 0", "y >= 0"], "x + y >= 0", True),
        (["x > 0"], "x >= 0", True),
        (["x = 0"], "x <= 0", True),
        (["z > 0"], "z >= 1", True),
        (["x > 0", "x >= 0 -> y >= 3"], "y >= 0", True),
        (["x >= 0", "y >= 0"], "x + y > 0", False),
        (["a == 0"], "b == 0", False),
    ],
)
def test_is_valid_implication(
    assumptions: list[str], conclusion: str, expected: bool
) -> None:
    result = why3.is_valid_implication(assumptions, conclusion)
    print(result)
    assert result == expected
