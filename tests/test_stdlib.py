# pyright: reportPrivateUsage=false

import typing
from collections.abc import Callable

from delphyne.stdlib import nodeclasses as nc


type _TestAlias[X] = list[X]


def test_type_annot_compatible():
    assert nc._type_annot_compatible(list[int], list)
    assert not nc._type_annot_compatible(list[int], dict)
    assert nc._type_annot_compatible(_TestAlias[int], _TestAlias)
    # No params are authorized on the right side!
    assert not nc._type_annot_compatible(_TestAlias[int], _TestAlias[int])


def test_decompose_callable_annot():
    decompose = nc._decompose_callable_annot
    assert decompose(Callable[[int, str], str]) == ([int, str], str)
    assert decompose(typing.Callable[[int, str], str]) == ([int, str], str)
    assert decompose(list[int]) == None
    assert decompose(Callable) == None
