"""
A strategy for finding invariants based on recursive abduction.

Used in the evaluation of the "Oracular Programming" paper:
https://arxiv.org/pdf/2502.05310
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import delphyne as dp
from delphyne import Branch, Compute, Fail, Strategy, strategy

import why3_utils as why3
from why3_utils import File, Formula

# fmt: off


#####
##### Strategy
#####


type _Proof = Sequence[why3.Formula]
"""
A proof is a sequence of established invariants
"""


type _Feedback = Sequence[why3.Obligation]
"""
A sequence of unproved obligations
"""


@strategy
def prove_program_by_recursive_abduction(
    prog: why3.File,
) -> Strategy[dp.Abduction, "ProveProgIP", why3.File]:
    invs = yield from dp.abduction(
        prove=lambda proved, goal:
            _prove_goal(prog, proved, goal)
                .using(lambda p: p.prove, ProveProgIP),
        suggest=lambda feedback:
            _suggest_invariants(feedback)
                .using(lambda p: p.suggest, ProveProgIP),
        search_equivalent=lambda proved, fml:
            _search_equivalent(proved, fml)
                .using(lambda p: p.search_equivalent, ProveProgIP),
        redundant=lambda proved, fml:
            _is_redundant(proved, fml)
                .using(lambda p: p.is_redundant, ProveProgIP),
    )
    return why3.add_invariants(prog, invs)


@strategy
def _prove_goal(
    prog: File, proved: Sequence[tuple[Formula, _Proof]], goal: Formula | None
) -> Strategy[Compute, None, dp.AbductionStatus[_Feedback, _Proof]]:
    invs = [p[0] for p in proved]
    aux_prog = _modified_program(prog, invs, goal)
    feedback = yield from dp.compute(why3.check)(aux_prog, aux_prog)
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
) -> Strategy[Branch | Fail, dp.PromptingPolicy, Sequence[Formula]]:
    assert len(unproved) > 0
    # We focus on the first unproved obligation
    answer = yield from dp.branch(
        SuggestInvariants(unproved[0]).using(dp.ambient_pp))
    return [s.invariant for s in answer.suggestions]


@strategy
def _search_equivalent(
    proved: Sequence[Formula], fml: Formula
) -> Strategy[Compute, None, Formula | None]:
    for p in proved:
        limpl = yield from dp.compute(why3.is_valid_implication)([p], fml)
        rimpl = yield from dp.compute(why3.is_valid_implication)([fml], p)
        if limpl and rimpl:
            return p
    return None


@strategy
def _is_redundant(
    proved: Sequence[Formula], fml: Formula
) -> Strategy[Compute, None, bool]:
    red = yield from dp.compute(why3.is_valid_implication)(proved, fml)
    return red


### Utilities


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


### Queries


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
    suggestions: Sequence["InvariantSuggestion"]


@dataclass
class SuggestInvariants(dp.Query[InvariantSuggestions]):
    unproved: why3.Obligation


#####
##### Policies
#####


@dataclass
class ProveProgIP:
    prove: dp.Policy[Compute, None]
    suggest: dp.Policy[Branch | Fail, dp.PromptingPolicy]
    is_redundant: dp.Policy[Compute, None]
    search_equivalent: dp.Policy[Compute, None]

    @staticmethod
    def make(
        pp: dp.PromptingPolicy,
        *,
        prove_timeout: float = 5.0,
        is_redundant_timeout: float = 1.0,
        search_equivalent_timeout: float = 0.3,
    ):
        """
        A convenience constructor that allows configuring timeouts for
        different operations.
        """

        def compute(timeout: float | None = None):
            sp = dp.dfs() @ dp.elim_compute(override_args={"timeout": timeout})
            return sp & None
        return ProveProgIP(
            prove= compute(timeout=prove_timeout),
            suggest=dp.dfs() & pp,
            search_equivalent=compute(timeout=search_equivalent_timeout),
            is_redundant=compute(timeout=is_redundant_timeout),
        )


@dp.ensure_compatible(prove_program_by_recursive_abduction)
def prove_program_by_saturation(
    *,
    model_name: str | None = None,
    model_cycle: Sequence[tuple[str, int]] | None = None,
    num_completions: int = 8,
    max_rollout_depth: int = 3,
    max_requests_per_attempt: int = 4,
    max_retries_per_step: int = 8,
    max_propagation_steps: int = 4,
    max_candidates: int = 12,
    max_proved: int = 8,
    temperature: float | None = None,
):
    """
    A policy that uses advanced saturation-based search, as featured and
    benchmarked the "Oracular Programming" paper.
    """

    if model_name:
        assert model_cycle is None
        model_cycle = [(model_name, 1)]
    assert model_cycle
    mcycle = [dp.standard_model(m) for (m, k) in model_cycle for _ in range(k)]

    def ip(model: dp.LLM):
        return ProveProgIP.make(
            dp.few_shot(
                model,
                temperature=temperature,
                num_completions=num_completions,
                max_requests=1))
    
    per_attempt = dp.BudgetLimit({dp.NUM_REQUESTS: max_requests_per_attempt})
    sp = dp.with_budget(per_attempt) @ dp.abduct_and_saturate(
        log_steps="info",
        max_rollout_depth=max_rollout_depth,
        max_raw_suggestions_per_step=3*num_completions,
        max_reattempted_candidates_per_propagation_step=max_retries_per_step,
        max_consecutive_propagation_steps=max_propagation_steps,
        remember_disproved=False,
        max_proved=max_proved,
        max_candidates=max_candidates,
    )
    return dp.sequence(sp & ip(m) for m in itertools.cycle(mcycle))


@dp.ensure_compatible(prove_program_by_recursive_abduction)
def prove_program_by_repeated_recursive_abduction(
    model_name: str,
    *,
    max_suggestions: int | None = 2,
    max_depth: int = 2,
    num_parallel: int = 1,
    temperature: float | None = None,
):
    """
    A simpler policy that can leverage parallelism.
    """
    ip = ProveProgIP.make(
        dp.few_shot(
            dp.standard_model(model_name),
            temperature=temperature,
            max_requests=1))
    sp = dp.abduct_recursively(
        max_suggestions=max_suggestions, max_depth=max_depth)
    return dp.loop() @ dp.parallel([sp for _ in range(num_parallel)]) & ip
