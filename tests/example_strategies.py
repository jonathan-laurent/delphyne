"""
Simple example strategies.

The strategies defined in this file are used to test `Tree` but also to
test the server (see `test_server`).
"""

from dataclasses import dataclass

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
        MakeSum(allowed, goal).search(lambda p: p.make_sum, MakeSumIP),
    )
    yield from dp.ensure(all(x in allowed for x in xs), "forbidden-num")
    yield from dp.ensure(sum(xs) == goal, "wrong-sum")
    return xs
