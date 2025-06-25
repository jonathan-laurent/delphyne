"""
A saturation-based strategy for finding invariants
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import delphyne as dp
from delphyne import Abduction, Branch, Computation, Failure, Strategy, strategy

import why3_utils as why3
from why3_utils import File, Formula

#####
##### The Main Query
#####


@dataclass
class InvariantSuggestions:
    obligation_kind: Literal["post", "init", "preserved"]
    obligation: str
    tricks: Sequence[str]
    suggestions: Sequence[why3.Formula]


@dataclass
class SuggestInvariants(dp.Query[InvariantSuggestions]):
    unproved: why3.Obligation


#####
##### Utilities and Main Strategy
#####


type _Proof = Sequence[why3.Formula]
"""
A proof is a sequence of established invariants
"""


type _Feedback = Sequence[why3.Obligation]
"""
A sequence of unproved obligations
"""


def _modified_program(prog: File, facts: Sequence[Formula], goal: Formula | None):
    """
    Get an updated version of the program where all proved invariants are added
    and only a specific goal annotation is left. Use `goal=None` if the goal is
    the final assertion.
    """
    invs = facts
    if goal is not None:
        invs = [*facts, goal]
    if invs:
        prog = why3.add_invariants(prog, invs)
    if goal is not None:
        prog, _ = why3.split_final_assertion(prog)
    return prog


@strategy
def _search_equivalent(
    proved: Sequence[Formula], fml: Formula
) -> Strategy[Computation, object, Formula | None]:
    for p in proved:
        limpl = yield from dp.compute(why3.is_valid_implication, [p], fml)
        rimpl = yield from dp.compute(why3.is_valid_implication, [fml], p)
        if limpl and rimpl:
            return p
    return None


@strategy
def _is_redundant(
    proved: Sequence[Formula], fml: Formula
) -> Strategy[Computation, object, bool]:
    red = yield from dp.compute(why3.is_valid_implication, proved, fml)
    return red


@strategy
def _prove_goal(
    prog: File, proved: Sequence[tuple[Formula, _Proof]], goal: Formula | None
) -> Strategy[Computation, object, dp.AbductionStatus[_Feedback, _Proof]]:
    invs = [p[0] for p in proved]
    aux_prog = _modified_program(prog, invs, goal)
    feedback = yield from dp.compute(why3.check, aux_prog, aux_prog)
    if feedback.success:
        return ("proved", invs)
    if feedback.error:
        return ("disproved", None)
    remaining = [o for o in feedback.obligations if not o.proved]
    assert len(remaining) > 0
    return ("feedback", remaining)


@strategy
def _suggest_invariants(
    unproved: Sequence[why3.Obligation],
) -> Strategy[Branch | Failure, dp.PromptingPolicy, Sequence[Formula]]:
    assert len(unproved) > 0
    # We focus on the first unproved obligation
    answer = yield from dp.branch(
        SuggestInvariants(unproved[0])(dp.PromptingPolicy, lambda p: p)
    )
    return answer.suggestions


@strategy
def prove_program_by_saturation(
    prog: why3.File,
) -> Strategy[Abduction, dp.PromptingPolicy, why3.File]:
    IP = dp.PromptingPolicy
    invs = yield from dp.abduction(
        prove=lambda proved, goal: _prove_goal(prog, proved, goal)(
            IP, lambda _: (dp.dfs() @ dp.elim_compute, None)
        ),
        suggest=lambda feedback: _suggest_invariants(feedback)(
            IP, lambda p: (dp.dfs(), p)
        ),
        search_equivalent=lambda proved, fml: _search_equivalent(proved, fml)(
            IP, lambda _: (dp.dfs() @ dp.elim_compute, None)
        ),
        redundant=lambda proved, fml: _is_redundant(proved, fml)(
            IP, lambda _: (dp.dfs() @ dp.elim_compute, None)
        ),
    )
    return why3.add_invariants(prog, invs)
