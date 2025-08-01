"""
Handling Feature Flags
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Never, cast

import delphyne.core as dp
import delphyne.core.inspect as insp
import delphyne.stdlib.policies as pol
from delphyne.stdlib.nodes import Branch, branch, spawn_node
from delphyne.stdlib.opaque import Opaque
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
        ans = self.flag_values()
        if answer.content not in ans:
            msg = f"Invalid flag value: '{answer.content}'."
            return dp.ParseError(description=msg)
        return cast(T, answer.content)


@dataclass(frozen=True)
class Flag[F: FlagQuery[Any]](dp.Node):
    flag: dp.TransparentQuery[Any]

    def summary_message(self) -> str:
        query = self.flag.attached.query
        assert isinstance(query, FlagQuery)
        name = query.name()
        return f"{name}: {', '.join(query.flag_values())}"

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
    query = dp.TransparentQuery.build(flag())
    ret = yield spawn_node(Flag, flag=query)
    return cast(T, ret)


@strategy
def variants[A: str, P, T](
    flag: type[FlagQuery[A]], alts: Callable[[A], Opaque[P, T]]
) -> dp.Strategy[Flag[Any] | Branch, P, T]:
    val = yield from get_flag(flag)
    return (yield from branch(alts(val)))


# Change: make all flags strings. The first flag is the default value.


@pol.contextual_tree_transformer
def elim_flag[F: FlagQuery[Any]](
    env: dp.PolicyEnv,
    policy: Any,
    flag: type[F],
    val: str,
) -> pol.PureTreeTransformerFn[Flag[F], Never]:
    assert val in flag.flag_values()

    def transform[N: dp.Node, P, T](
        tree: dp.Tree[Flag[F] | N, P, T],
    ) -> dp.Tree[N, P, T]:
        if isinstance(tree.node, Flag):
            tree = cast(dp.Tree[Any, P, T], tree)
            node = cast(Flag[Any], tree.node)
            if isinstance(query := node.flag.attached.query, flag):
                node = cast(Flag[F], node)
                answer = dp.Answer(None, val)
                assert answer in query.finite_answer_set()
                tracked = node.flag.attached.parse_answer(answer)
                assert not isinstance(tracked, dp.ParseError)
                return transform(tree.child(tracked))
        tree = cast(dp.Tree[Any, P, T], tree)
        node = cast(Any, tree.node)
        return tree.transform(node, transform)

    return transform
