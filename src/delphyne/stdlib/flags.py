"""
The `Flag` Effect
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Never, cast

import delphyne.core.inspect as insp
import delphyne.core_and_base as dp
from delphyne.core_and_base import PolicyEnv, spawn_node
from delphyne.stdlib.queries import Query


class FlagQuery[T: str](Query[T]):
    """
    Base class for flag queries.

    Type parameter `T` must be of the form `Literal[s1, ..., sn]` where
    `si` are string literals. The first value is considered the default.
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
    """
    The standard `Flag` effect.

    Flags allow providing several implementations for a strategy
    component, and have policies select which variant to use (or perform
    search at runtime for selecting variants).

    For each flag, a subclass of `FlagQuery` must be defined, which
    admits a finite set of answers (one per allowed flag value), along
    with a default answer. Type parameter `F` denotes the type of the
    flag query that can be issued. To express a signature wih several
    flag queries, use `Flag[A] | Flag[B]` instead of `Flag[A | B]`, so
    that both kinds of flags can be eliminated separately.

    ??? info "Behavior in demonstrations"
        Because flag queries override `AbstractQuery.default_answer`,
        default flag values are automatically selected by the
        demonstration interpreter. This behaviour can be overriden by
        adding answers for flag queries in the `queries` section, or by
        using value-based hints (i.e., `#flag_value`, which is possible
        since flag queries implement `AbstractQuery.finite_answer_set`).
    """

    flag: dp.TransparentQuery[Any]

    def summary_message(self) -> str:
        query = self.flag.attached.query
        assert isinstance(query, FlagQuery)
        name = query.query_name()
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
    """
    Triggering function for the `Flag` effect.

    Takes a flag query type as an argument and return a flag value.

    !!! info
        A more precise type cannot be given for this function since
        Python does not support higher-kinded types.
    """
    query = dp.TransparentQuery.build(flag())
    ret = yield spawn_node(Flag, flag=query)
    return cast(T, ret)


@dp.contextual_tree_transformer
def elim_flag[F: FlagQuery[Any]](
    env: PolicyEnv,
    policy: Any,
    flag: type[F],
    val: str,
) -> dp.PureTreeTransformerFn[Flag[F], Never]:
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
