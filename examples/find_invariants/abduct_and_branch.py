"""
A simple Delphyne strategy to discover loop invariants with Why3.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import delphyne as dp
from delphyne import Branch, Compute, Fail, Strategy, Value, strategy

import why3_utils as why3

# fmt: off


#####
##### Inner Policy Types
#####


@dataclass
class ProposeInvsIP:
    propose: dp.PromptingPolicy
    novel: dp.PromptingPolicy


@dataclass
class ProveProgIP:
    propose: "dp.Policy[Branch | Fail, ProposeInvsIP]"
    eval: dp.PromptingPolicy
    quantify_eval: "Callable[[ProofStateMetrics], float] | None"


#####
##### Type Definitions
#####


@dataclass
class ProofStateMetrics:
    has_redundant_invs: bool


type Proposal = Sequence[why3.Formula]


type Blacklist = Sequence[Proposal]



#####
##### Strategies
#####


@strategy
def prove_program_via_abduction_and_branching(
    prog: why3.File,
) -> Strategy[Branch | Value | Fail | Compute, ProveProgIP, why3.File]:
    annotated: why3.File = prog
    while True:
        feedback = yield from dp.compute(why3.check)(prog, annotated)
        yield from dp.ensure(feedback.error is None, "invalid program")
        if feedback.success:
            return annotated
        remaining = [o for o in feedback.obligations if not o.proved]
        yield from dp.ensure(
            len(remaining) == 1, "too many remaining obligations")
        unproved = remaining[0]
        yield from dp.ensure(
            not why3.invariant_init_obligation(unproved), "init")
        if annotated != prog:  # if invariant annotations are present
            yield from dp.value(
                EvaluateProofState(unproved)
                    .using(lambda p: p.eval, ProveProgIP),
                lambda p: p.quantify_eval)
        new_invariants = yield from dp.branch(
            dp.iterate(
                lambda prior:
                    propose_invariants(unproved, prior)
                        .using(lambda p: p.propose, ProveProgIP)))
        annotated = why3.add_invariants(annotated, new_invariants)


@strategy
def propose_invariants(
    obligation: why3.Obligation,
    blacklist: Sequence[Proposal] | None,
) -> Strategy[Branch | Fail, ProposeInvsIP, tuple[Proposal, Blacklist]]:
    if blacklist is None:
        blacklist = []
    proposal = yield from dp.branch(
        ProposeInvariants(obligation, blacklist)
            .using(lambda p: p.propose, ProposeInvsIP))
    sanity_check = all(why3.no_invalid_formula_symbol(inv) for inv in proposal)
    yield from dp.ensure(sanity_check, "sanity check failed")
    if blacklist:
        novel = yield from dp.branch(
            IsProposalNovel(proposal, blacklist)
                .using(lambda p: p.novel, ProposeInvsIP))
        yield from dp.ensure(novel, "proposal is not novel")
    return proposal, [*blacklist, proposal]


#####
##### Queries
#####


@dataclass
class ProposeInvariants(dp.Query[Proposal]):
    unproved: why3.Obligation
    blacklist: Sequence[Proposal]

    __parser__ = dp.last_code_block.yaml


@dataclass
class IsProposalNovel(dp.Query[bool]):
    proposal: Proposal
    blacklist: Sequence[Proposal]

    __parser__ = dp.last_code_block.yaml


@dataclass
class EvaluateProofState(dp.Query[ProofStateMetrics]):
    unproved: why3.Obligation

    __parser__ = dp.last_code_block.yaml


#####
##### Policies
#####


def prove_program_via_abduction_and_branching_policy(
    fancy_model: str = "gpt-4o",
    base_model: str = "gpt-4o",
    max_depth: int = 2,
    max_requests_per_proposal: int = 5,
    root_proposal_penalty: float = 0.7,
    nonroot_proposal_penalty: float = 1.0,
    max_nonroot_proposals: int = 3,
    enable_state_evaluation: bool = False,
    min_value: float = 0.3,
) -> dp.Policy[Branch | Value | Fail | Compute, ProveProgIP]:

    def compute_value(metrics: ProofStateMetrics) -> float:
        prob = 1
        if metrics.has_redundant_invs:
            prob = 0
        return max(min_value, prob)

    def child_confidence_prior(depth: int, prev_gen: int) -> float:
        if depth >= 1:
            if prev_gen >= max_nonroot_proposals:
                return 0
            return nonroot_proposal_penalty ** prev_gen
        return root_proposal_penalty ** prev_gen

    n = dp.NUM_REQUESTS
    fancy = dp.openai_model(fancy_model)
    base = dp.openai_model(base_model)

    propose_ip = ProposeInvsIP(
        propose=dp.few_shot(fancy),
        novel=dp.take(1) @ dp.few_shot(base),
    )
    proposal_limit = dp.BudgetLimit({n: max_requests_per_proposal})
    prove_prog_ip = ProveProgIP(
        propose=dp.with_budget(proposal_limit) @ dp.dfs() & propose_ip,
        eval=dp.take(1) @ dp.few_shot(base),
        quantify_eval=compute_value if enable_state_evaluation else None,
    )
    bestfs = dp.best_first_search(
        child_confidence_prior=child_confidence_prior,
        max_depth=max_depth)

    return bestfs @ dp.elim_compute() & prove_prog_ip
