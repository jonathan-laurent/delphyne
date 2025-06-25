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


PROG_1 = """
use int.Int

let main () diverges =
  let ref x = any int in
  let ref y = any int in
  x <- 1;
  y <- 0;
  while y < 100000 do
    x <- x + y;
    y <- y + 1
  done;
  assert { x >= y }
"""

PROG_2 = """
use int.Int

let main () diverges =
  let ref n = any int in
  let ref y = any int in
  let ref x = 1 in
  while x <= n do
    y <- n - x;
    x <- x + 1
  done;
  if n > 0 then
    assert { y <= n }
"""


@pytest.mark.parametrize(
    "prog",
    [PROG_1, PROG_2],
)
def test_split_restore_final_assertion_inverse(prog: str) -> None:
    """
    split_final_assertion and restore_final_assertion are inverse operations.
    """
    modified_prog, extracted_formula = why3.split_final_assertion(prog)
    restored_prog = why3.restore_final_assertion(modified_prog, extracted_formula)
    assert restored_prog == prog
    assert "assert { true }" in modified_prog
    assert extracted_formula.strip() != ""
    assert extracted_formula.strip() != "true"
