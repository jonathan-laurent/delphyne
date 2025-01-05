"""
Simple example strategies.

The strategies defined in this file are used to test `Tree` but also to
test the server (see `test_server`).
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence, cast

import delphyne as dp

#####
##### MakeSum
#####


@dataclass
class MakeSum(dp.Query[list[int]]):
    allowed: list[int]
    goal: int

    @classmethod
    def modes(cls) -> dp.Modes[list[int]]:
        return {None: dp.AnswerMode(dp.raw_yaml)}


@dataclass
class MakeSumIP:
    make_sum: dp.PromptingPolicy


@dp.strategy
def make_sum(
    allowed: list[int], goal: int
) -> dp.Strategy[dp.Branch | dp.Failure, MakeSumIP, list[int]]:
    xs = yield from dp.branch(
        cands=MakeSum(allowed, goal).search(lambda p: p.make_sum, MakeSumIP),
    )
    yield from dp.ensure(all(x in allowed for x in xs), "forbidden-num")
    yield from dp.ensure(sum(xs) == goal, "wrong-sum")
    return xs


#####
##### Adding support for conjecture nodes
#####


@dataclass(frozen=True)
class Conjecture(dp.Node):
    candidate: dp.OpaqueSpace[Any, Any]
    disprove: Callable[[dp.Tracked[Any]], dp.OpaqueSpace[Any, None]]
    aggregate: Callable[
        [tuple[dp.Tracked[Any], ...]], dp.OpaqueSpace[Any, Sequence[Any]]
    ]

    def navigate(self) -> dp.Navigation:
        return (yield self.candidate)


def make_conjecture[P, T](
    candidate: dp.OpaqueSpace[P, T],
    disprove: Callable[[T], dp.OpaqueSpace[P, None]],
    aggregate: Callable[[tuple[T, ...]], dp.OpaqueSpace[P, Sequence[T]]],
) -> dp.Strategy[Conjecture, P, T]:
    cand = yield dp.spawn_node(
        Conjecture, candidate=candidate, disprove=disprove, aggregate=aggregate
    )
    return cast(T, cand)


@dp.search_policy
async def just_guess[P, T](
    tree: dp.Tree[Conjecture, P, T], env: dp.PolicyEnv, policy: P
) -> dp.Stream[T]:
    """
    Do a DFS, treating conjecture nodes as simple branching nodes.
    """
    match tree.node:
        case dp.Success(x):
            yield dp.Yield(x)
        case Conjecture(candidate):
            async for msg in dp.bind_stream(
                candidate.stream(env, policy),
                lambda y: just_guess()(tree.child(y), env, policy),
            ):
                yield msg
