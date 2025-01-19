"""
A simple Delphyne strategy to discover loop invariants with Why3.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import delphyne as dp
from delphyne import Branch, Computation, Failure, Strategy, Value, strategy

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
    propose: "dp.Policy[Branch | Failure, ProposeInvsIP]"
    eval: dp.PromptingPolicy
    quantify_eval: "Callable[[ProofStateMetrics], float]"


#####
##### Type Definitions
#####


@dataclass
class UnprovedObligation:
    obligation_name: str
    relevance_hints: str


@dataclass
class ProofStateMetrics:
    prob_incorrect: float
    prob_redundant: float


type Proposal = Sequence[why3.Formula]


#####
##### Strategies
#####


@strategy
def prove_program(
    prog: why3.File,
) -> Strategy[Branch | Value | Failure | Computation, ProveProgIP, why3.File]:
    annotated: why3.File = prog
    while True:
        feedback = yield from dp.compute(why3.check, prog, annotated)
        yield from dp.ensure(feedback.error is None, "invalid program")
        if feedback.success:
            return annotated
        remaining = [o for o in feedback.obligations if not o.proved]
        yield from dp.ensure(len(remaining) == 1, "too many remaining obligations")
        unproved = UnprovedObligation(remaining[0].name, remaining[0].relevance_hints)
        if annotated != prog:
            yield from dp.value(
                EvaluateProofState(unproved)(ProveProgIP, lambda p: p.eval),
                lambda p: p.quantify_eval)
        new_invariants = yield from dp.branch(
            dp.iterate(
                lambda prior: propose_invariants(unproved, prior)(
                    ProveProgIP, lambda p: p.propose)))
        annotated = why3.add_invariants(annotated, new_invariants)


@strategy
def propose_invariants(
    obligation: UnprovedObligation,
    blacklist: Sequence[Proposal] | None,
) -> Strategy[Branch | Failure, ProposeInvsIP, tuple[Proposal, Sequence[Proposal]]]:
    if blacklist is None:
        blacklist = []
    proposal = yield from dp.branch(
        ProposeInvariants(obligation, blacklist)(ProposeInvsIP, lambda p: p.propose))
    sanity_check = all(why3.no_invalid_formula_symbol(inv) for inv in proposal)
    yield from dp.ensure(sanity_check, "sanity check failed")
    if blacklist:
        novel = yield from dp.branch(
            IsProposalNovel(proposal, blacklist)(ProposeInvsIP, lambda p: p.novel))
        yield from dp.ensure(novel, "proposal is not novel")
    return proposal, [*blacklist, proposal]


#####
##### Queries
#####


@dataclass
class ProposeInvariants(dp.Query[Proposal]):
    unproved: UnprovedObligation
    blacklist: Sequence[Proposal]

    __parser__: ClassVar = dp.yaml_from_last_block


@dataclass
class IsProposalNovel(dp.Query[bool]):
    proposal: Proposal
    blacklist: Sequence[Proposal]

    __parser__: ClassVar = dp.yaml_from_last_block


@dataclass
class EvaluateProofState(dp.Query[ProofStateMetrics]):
    unproved: UnprovedObligation

    def parse(self, mode: str | None, answer: str) -> ProofStateMetrics:
        metrics = dp.yaml_from_last_block(ProofStateMetrics, answer)
        redundant_ok = 0 <= metrics.prob_redundant <= 1
        incorrect_ok = 0 <= metrics.prob_incorrect <= 1
        if not (redundant_ok and incorrect_ok) >= 1:
            raise dp.ParseError("Invalid metrics.")
        return metrics


#####
##### Policies
#####


def prove_program_policy(
    fancy_model: str = "gpt-4o",
    base_model: str = "gpt-4o",
    max_depth: int = 2,
    min_value: float = 0.2,
    max_requests_per_proposal: int = 10,
    proposal_penalty: float = 0.7,
    max_deep_proposals: int = 2,
    penalize_redundancy: bool = True,
) -> dp.Policy[Branch | Value | Failure | Computation, ProveProgIP]:

    def compute_value(metrics: ProofStateMetrics) -> float:
        prob = (1 - metrics.prob_incorrect)
        if penalize_redundancy:
            prob *= (1 - metrics.prob_redundant)
        return max(min_value, prob)

    def child_confidence_prior(depth: int, prev_gen: int) -> float:
        if depth >= 1 and prev_gen >= max_deep_proposals:
            return 0
        return proposal_penalty ** prev_gen

    n = dp.NUM_REQUESTS
    fancy = dp.openai_model(fancy_model)
    base = dp.openai_model(base_model)

    propose_ip = ProposeInvsIP(
        propose=dp.few_shot(fancy),
        novel=dp.take(1) @ dp.few_shot(base),
    )
    proposal_limit = dp.BudgetLimit({n: max_requests_per_proposal})
    prove_prog_ip = ProveProgIP(
        propose=(dp.with_budget(proposal_limit) @ dp.dfs(), propose_ip),
        eval=dp.take(1) @ dp.few_shot(base),
        quantify_eval=compute_value,
    )
    bestfs = dp.best_first_search(
        child_confidence_prior=child_confidence_prior,
        max_depth=max_depth)

    return (bestfs @ dp.elim_compute, prove_prog_ip)
