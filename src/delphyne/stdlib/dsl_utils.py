"""
Extension of the standard library for Delphyne.
"""

from typing import Callable

from delphyne.stdlib.nodes import Failure, Strategy, fail


def ensure_noexcept[T](f: Callable[[], T]) -> Strategy[Failure, T]:
    try:
        return f()
    except Exception as e:
        return (yield from fail("An exception occurred: " + str(e)))


def ensure_some[T](x: T | None, msg: str = "") -> Strategy[Failure, T]:
    if x is None:
        return (yield from fail(msg))
    else:
        return x
