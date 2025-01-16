# pyright: basic

import typing
from collections.abc import Callable
from typing import Any, Sequence

import pytest

import delphyne.core.inspect as insp
import delphyne.core.node_fields as nf


def test_function_args_dict():
    def f(x, y, z=0):
        pass

    assert insp.function_args_dict(f, (1, 2), {}) == {"x": 1, "y": 2}


def test_type_annot_compatible():
    assert nf._type_annot_compatible(list[int], list)
    assert not nf._type_annot_compatible(list[int], dict)


def test_decompose_callable_annot():
    decompose = nf._decompose_callable_annot
    assert decompose(Callable[[int, str], str]) == ([int, str], str)
    assert decompose(typing.Callable[[int, str], str]) == ([int, str], str)
    assert decompose(list[int]) is None
    assert decompose(Callable) is None


@pytest.mark.parametrize(
    "inp, i, out",
    [
        (list[int], 0, int),
        (list[tuple[int, float]], 1, tuple[int, float]),
        (Sequence[bool], 0, bool),
        (tuple[int, ...], 0, int),
        (tuple[int, ...], 1, int),
        (tuple[int, bool], 0, int),
        (tuple[int, bool], 1, bool),
        (tuple[int, bool, float], 2, float),
    ],
)
def test_element_type_of_sequence_type(inp: Any, i, out: Any):
    assert insp.element_type_of_sequence_type(inp, i) == out
