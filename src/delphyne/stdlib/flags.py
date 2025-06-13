"""
Handling Feature Flags
"""

from collections.abc import Sequence
from typing import cast

import delphyne.core.inspect as insp
from delphyne.stdlib.queries import Query


class FlagQuery[T](Query[T]):
    """
    Base class for flag queries. T must be of the form `Literal[None,
    s1, ..., sn]` where `si` are string literals.
    """

    def finite_answer_set(self) -> Sequence[T] | None:
        ans = self.answer_type()
        assert (args := insp.literal_type_args(ans)) is not None
        assert len(args) > 0
        assert all(a is None or isinstance(a, str) for a in args)
        return cast(Sequence[T], args)

    def default_answer(self) -> T:
        ans = self.finite_answer_set()
        assert ans is not None and None in ans
        return cast(T, None)
