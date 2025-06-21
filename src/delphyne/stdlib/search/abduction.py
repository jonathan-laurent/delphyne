"""
Strategies and Policies for Recursive Abduction.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal, cast

import delphyne.core as dp
from delphyne.stdlib.nodes import spawn_node
from delphyne.stdlib.policies import search_policy
from delphyne.stdlib.streams import take_one

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


@dataclass
class _CandInfo:
    feedback: _Feedback
    num_proposed: float
    num_visited: float


class _Abort(Exception): ...


@search_policy
def abduct_and_saturate[P, Proof](
    tree: dp.Tree[Abduction, P, Proof],
    env: dp.PolicyEnv,
    policy: P,
) -> dp.Stream[Proof]:
    """ """
    # TODO: finish this. We have to be careful about everything being tracked.
    # We should have tracked and untracked facts everywhere...
    # All `facts` and `proof` objects are tracked
    facts: list[tuple[_Fact, _Proof]] = []
    candidates: dict[_Fact, _CandInfo] = {}
    proved: set[_Fact] = set()  # redundant with `facts` but convenient
    disproved: set[_Fact] = set()
    _redundant: set[_Fact] = set()
    # maps a candidate to the canonical representative found in
    # `candidates` or `disproved`.
    _equivalent: dict[_Fact, _Fact] = {}

    assert isinstance(tree.node, Abduction)
    node = tree.node

    def register_fact(  # pyright: ignore[reportUnusedFunction]
        fact: _Fact, proof: _Proof
    ) -> dp.Stream[None]:
        # We must re-examine all candidates
        facts.append((fact, proof))
        proved.add(fact)
        del candidates[fact]
        old_candidates = candidates.copy()
        candidates.clear()
        newly_proved: list[tuple[_Fact, _Proof]] = []
        for c, _i in old_candidates.items():
            assert isinstance(tree.node, Abduction)
            pstream = node.prove(fact, c).stream(env, policy)
            res = yield from take_one(pstream)
            if res is None:
                raise _Abort()
            status, payload = res[0], res[1]
            if status.value == "disproved":
                disproved.add(c)
            elif status.value == "proved":
                newly_proved.append((c, payload))
            else:
                # TODO: put back in candidates
                pass
            pass
        pass

    assert False
