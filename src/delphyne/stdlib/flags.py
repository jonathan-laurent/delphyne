"""
Handling Feature Flags
"""

from collections.abc import Callable, Sequence
from typing import Any, Never, cast

from why3py import dataclass

import delphyne.core as dp
import delphyne.core.inspect as insp
from delphyne.stdlib.nodes import Branch, branch, spawn_node
from delphyne.stdlib.policies import ContextualTreeTransformer
from delphyne.stdlib.queries import Query
from delphyne.stdlib.strategies import strategy


class FlagQuery[T: str](Query[T]):
    """
    Base class for flag queries. T must be of the form `Literal[s1, ...,
    sn]` where `si` are string literals. The first value is considered
    the default.
    """

    @classmethod
    def flag_values(cls) -> Sequence[str]:
        ans = insp.first_parameter_of_base_class(cls)
        assert (args := insp.literal_type_args(ans)) is not None
        assert len(args) > 0
        assert all(isinstance(a, str) for a in args)
        return cast(Sequence[str], args)

    def finite_answer_set(self) -> Sequence[dp.Answer]:
        vals = self.flag_values()
        return [dp.Answer(None, v) for v in vals]

    def default_answer(self) -> dp.Answer:
        return self.finite_answer_set()[0]

    def parse_answer(self, answer: dp.Answer) -> T | dp.ParseError:
        if not isinstance(answer.content, str):
            return dp.ParseError(description="Flag answer must be a string.")
        ans = self.finite_answer_set()
        assert ans is not None
        if answer.content not in ans:
            msg = f"Invalid flag value: {answer.content}."
            return dp.ParseError(description=msg)
        return cast(T, answer.content)


@dataclass(frozen=True)
class Flag[F: FlagQuery[Any]](dp.Node):
    flag: dp.AttachedQuery[Any]

    def summary_message(self) -> str:
        query = self.flag.query
        assert isinstance(query, FlagQuery)
        name = query.name()
        return f"{name}: {query.flag_values()}"

    def navigate(self) -> dp.Navigation:
        return (yield self.flag)


# We cannot give a more precise type signature here because Python does
# not support higher-kinded type vars. Ideally, we would write:
#
#     def query_flag[T, F: FlagQuery[T]](...) -> ...:
#
# In the meantime, we just do not constrain F.


def get_flag[T: str](
    flag: type[FlagQuery[T]],
) -> dp.Strategy[Flag[Any], object, T]:
    query = dp.AttachedQuery.build(flag())
    ret = yield spawn_node(Flag, flag=query)
    return cast(T, ret)


@strategy
def variants[A: str, P, T](
    flag: type[FlagQuery[A]], alts: Callable[[A], dp.OpaqueSpaceBuilder[P, T]]
) -> dp.Strategy[Flag[Any] | Branch, P, T]:
    val = yield from get_flag(flag)
    return (yield from branch(alts(val)))


# Change: make all flags strings. The first flag is the default value.


def pure_elim_flag[F: FlagQuery[Any], N: dp.Node, P, T](
    flag: type[F],
    val: str,
    tree: dp.Tree[Flag[F] | N, P, T],
) -> dp.Tree[N, P, T]:
    if isinstance(tree.node, Flag):
        node = cast(Flag[Any], tree.node)  # type: ignore
        if isinstance(query := node.flag.query, flag):
            node = cast(Flag[F], node)
            answer = dp.Answer(None, val)
            assert answer in query.finite_answer_set()
            tracked = node.flag.parse_answer(answer)
            assert not isinstance(tracked, dp.ParseError)
            return pure_elim_flag(flag, val, tree.child(tracked))
    return tree.transform(tree.node, lambda n: pure_elim_flag(flag, val, n))  # type: ignore


def elim_flag[F: FlagQuery[Any]](
    flag: type[F], val: str
) -> ContextualTreeTransformer[Flag[F], Never]:
    assert val in flag.flag_values()
    return ContextualTreeTransformer[Flag[F], Never].pure(
        lambda tree: pure_elim_flag(flag, val, tree)  # type: ignore
    )
