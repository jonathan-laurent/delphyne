"""
A saturation-based strategy for finding invariants
"""

from collections.abc import Sequence
from dataclasses import dataclass

import delphyne as dp
from delphyne import Branch, Computation, Failure, Strategy, strategy

import why3_utils as why3
from why3_utils import File, Formula

# fmt: off


#####
##### The Main Query
#####



@dataclass
class InvariantSuggestion:
    trick: str
    invariant: str


@dataclass
class InvariantSuggestions:
    """
    A sequence of suggestions, each of which consists in a trick name
    along with an invariant proposal.
    """
    suggestions: Sequence[InvariantSuggestion]


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


def _modified_program(
    prog: File, facts: Sequence[Formula], goal: Formula | None
) -> File:
    """
    Get an updated version of the program where all proved invariants
    are added and only a specific goal annotation is left. Use
    `goal=None` if the goal is the final assertion.
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
        SuggestInvariants(unproved[0]).using(dp.ambient_pp))
    return [s.invariant for s in answer.suggestions]


@strategy
def prove_program_by_saturation(
    prog: why3.File,
) -> Strategy[dp.Abduction, dp.PromptingPolicy, why3.File]:
    invs = yield from dp.abduction(
        prove=lambda proved, goal:
            _prove_goal(prog, proved, goal).using(dp.just_compute),
        suggest=lambda feedback:
            _suggest_invariants(feedback).using(dp.just_dfs),
        search_equivalent=lambda proved, fml:
            _search_equivalent(proved, fml).using(dp.just_compute),
        redundant=lambda proved, fml:
          _is_redundant(proved, fml).using(dp.just_compute),
        inner_policy_type=dp.PromptingPolicy
    )
    return why3.add_invariants(prog, invs)


#####
##### Policy
#####


def prove_program_by_saturation_policy(
    model_name: dp.StandardModelName,
    num_concurrent: int = 3,
    max_rollout_depth: int = 3,
    temperature: float | None = None
) -> dp.Policy[dp.Abduction, dp.PromptingPolicy]:
    model = dp.standard_model(model_name)
    pp = dp.few_shot(
        model,
        temperature=temperature,
        num_concurrent=num_concurrent,
        max_requests=1)
    sp = dp.abduct_and_saturate(
        verbose=True, max_rollout_depth=max_rollout_depth)
    return (sp, pp)
