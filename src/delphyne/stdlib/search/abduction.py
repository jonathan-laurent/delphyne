"""
Strategies and Policies for Recursive Abduction.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import delphyne.core as dp
from delphyne.stdlib.nodes import spawn_node

# For readability of the `Abduction` definition
type _Fact = Any
type _Proof = Any
type _Feedback = Any
type _Status = Any


@dataclass
class Abduction(dp.Node):
    """
    Node for the singleton tree produced by `abduction`.
    See `abduction` for details.

    An action is a successful proof of the main goal.
    """

    prove: Callable[
        [Sequence[tuple[_Fact, _Proof]], _Fact | None],
        dp.OpaqueSpace[Any, _Status],
    ]
    suggest: Callable[
        [_Feedback],
        dp.OpaqueSpace[Any, Sequence[_Fact]],
    ]
    search_equivalent: Callable[
        [Sequence[_Fact], _Fact],
        dp.OpaqueSpace[Any, _Fact | None],
    ]
    redundant: Callable[
        [Sequence[_Fact], _Fact],
        dp.OpaqueSpace[Any, bool],
    ]

    def navigate(self) -> dp.Navigation:
        def aux(fact: _Fact | None) -> dp.Navigation:
            res = yield self.prove([], fact)
            status, payload = res[0], res[1]
            if status.value == "proved":
                return payload
            elif status.value == "disproved":
                assert False
            else:
                assert status.value == "feedback"
                feedback = payload
                suggestions = yield self.suggest(feedback)
                proved: list[Any] = []
                for s in suggestions:
                    proved.append((s, (yield from aux(s))))
                res = yield self.prove(proved, fact)
                status, payload = res[0], res[1]
                assert status.value == "proved"
                return payload

        return (yield from aux(None))


type AbductionStatus[Feedback, Proof] = (
    tuple[Literal["disproved"], None]
    | tuple[Literal["proved"], Proof]
    | tuple[Literal["feedback"], Feedback]
)


def abduction[Fact, Feedback, Proof, P](
    prove: Callable[
        [Sequence[tuple[Fact, Proof]], Fact | None],
        dp.OpaqueSpaceBuilder[P, AbductionStatus[Feedback, Proof]],
    ],
    suggest: Callable[
        [Feedback],
        dp.OpaqueSpaceBuilder[P, Sequence[Fact]],
    ],
    search_equivalent: Callable[
        [Sequence[Fact], Fact], dp.OpaqueSpaceBuilder[P, Fact | None]
    ],
    redundant: Callable[
        [Sequence[Fact], Fact], dp.OpaqueSpaceBuilder[P, bool]
    ],
) -> dp.Strategy[Abduction, P, Proof]:
    """
    Higher-order strategy for proving a fact via recursive abduction.

    Arguments:

      prove: take a sequence of already established facts as an
        argument along with a new fact, and attempt to prove this new
        fact. Three outcomes are possible: the fact is proved,
        disproved, or a list of suggestions are made that might be
        helpful to prove first. `None` denotes the top-level goal to be
        proved.

      suggest: take some feedback from the `prove` function and return a
        sequence of fact candidates that may be useful to prove before
        reattempting the original proof.

      search_equivalent: take a collection of facts along with a new
        one, and return either the first fact of the list equivalent to
        the new fact or `None`. This is used to avoid spending search in
        proving equivalent facts.

      redundant: take a collection of established facts and decide
        whether they imply a new fact candidate. This is useful to avoid
        trying to prove and accumulating redundant facts.

    Returns:
      a proof of the top-level goal.
    """
    res = yield spawn_node(
        Abduction,
        prove=prove,
        suggest=suggest,
        search_equivalent=search_equivalent,
        redundant=redundant,
    )
    return cast(Proof, res)
