# pyright: basic

import typing
from collections.abc import Callable

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
