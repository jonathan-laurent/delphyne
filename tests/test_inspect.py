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


@pytest.mark.parametrize(
    "inp, out",
    [
        (typing.Never, []),
        (typing.Union[int, str], [int, str]),
        (int | str, [int, str]),
        (int | (str | bool), [int, str, bool]),
        (typing.Union[int, bool | float], [int, bool, float]),
    ],
)
def test_union_components(inp: Any, out: Any):
    assert list(insp.union_components(inp)) == list(out)


def test_is_sequence_type():
    assert insp.is_sequence_type(list[int])
    assert insp.is_sequence_type(typing.Sequence[int])
    assert insp.is_sequence_type(tuple[int, ...])
    assert not insp.is_sequence_type(tuple[int, str])
    assert not insp.is_sequence_type(dict[str, int])


type _TestLitType = typing.Literal[1, 2, 3]


@pytest.mark.parametrize(
    "inp, out",
    [
        (typing.Literal[1, 2, 3], [1, 2, 3]),
        (typing.Literal["a"], ["a"]),
        (typing.Literal[True, False], [True, False]),
        (typing.Literal[1, "a"], [1, "a"]),
        (_TestLitType, [1, 2, 3]),
        (int, None),
    ],
)
def test_literal_type_args(inp, out):
    res = insp.literal_type_args(inp)
    res = list(res) if res is not None else None
    assert res == out


def test_is_method_overridden():
    class Base:
        def method(self):
            pass

    class Derived(Base):
        def method(self):
            pass

    class NotOverridden(Base):
        pass

    assert insp.is_method_overridden(Base, Derived, "method")
    assert not insp.is_method_overridden(Base, NotOverridden, "method")
