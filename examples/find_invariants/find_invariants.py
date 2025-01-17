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
class ProposeChangeIP:
    propose: dp.PromptingPolicy
    eval: dp.PromptingPolicy


@dataclass
class ProveProgramIP:
    propose_change: "dp.Policy[Branch | Failure, ProposeChangeIP]"
    eval_proposal: dp.PromptingPolicy
    quantify_eval: "Callable[[ProofStateMetrics], float]"


#####
##### Type Definitions
#####


@dataclass
class UnprovedObligation:
    obligation_name: str
    relevance_hints: str


@dataclass
class Proposal:
    change_summary: str
    annotated: why3.File

    def __str__(self) -> str:
        return self.change_summary


@dataclass
class ProofStateMetrics:
    num_annotations: int


@dataclass
class ProposalEvaluation:
    change_summary: str
    good_proposal: bool
    comment: str


#####
##### Strategies and Queries
#####


@strategy
def prove_program(
    prog: why3.File,
) -> Strategy[Branch | Value | Failure | Computation, ProveProgramIP, why3.File]:
    annotated: why3.File = prog
    while True:
        feedback = yield from dp.compute(why3.check, prog, annotated)
        yield from dp.ensure(feedback.error is None, "invalid program")
        if feedback.success:
            return annotated
        remaining = [o for o in feedback.obligations if not o.proved]
        yield from dp.ensure(len(remaining) == 1, "too many remaining obligations")
        unproved = UnprovedObligation(remaining[0].name, remaining[0].relevance_hints)
        yield from dp.value(
            EvaluateProofState(unproved)(ProveProgramIP, lambda p: p.eval_proposal),
            lambda p: p.quantify_eval)
        proposal = yield from dp.branch(
            dp.iterate(
                lambda prior: propose_change(annotated, unproved, prior)(
                    ProveProgramIP, lambda p: p.propose_change)))
        annotated = proposal.annotated


@dataclass
class EvaluateProofState(dp.Query[ProofStateMetrics]):
    unproved: UnprovedObligation

    def parse(self, mode: str | None, answer: str) -> ProofStateMetrics:
        metrics = dp.yaml_from_last_block(ProofStateMetrics, answer)
        if not metrics.num_annotations >= 1:
            raise dp.ParseError("Invalid metrics.")
        return metrics


@strategy
def propose_change(
    annotated: why3.File,
    unproved: UnprovedObligation,
    prior_changes: Sequence[str] | None,
) -> Strategy[Branch | Failure, ProposeChangeIP, tuple[Proposal, Sequence[str]]]:
    if prior_changes is None:
        prior_changes = []
    new_annotated = yield from dp.branch(
        ProposeChange(unproved, prior_changes)(ProposeChangeIP, lambda p: p.propose))
    evaluation = yield from dp.branch(
        EvaluateProposal(annotated, new_annotated, prior_changes)(
            ProposeChangeIP, lambda p: p.eval))
    yield from dp.ensure(evaluation.good_proposal)
    proposal = Proposal(evaluation.change_summary, new_annotated)
    return proposal, [*prior_changes, proposal.change_summary]



@dataclass
class ProposeChange(dp.Query[why3.File]):
    unproved: UnprovedObligation
    prior_changes_summaries: Sequence[str]

    __parser__: ClassVar = dp.string_from_last_block


@dataclass
class EvaluateProposal(dp.Query[ProposalEvaluation]):
    before: why3.File
    after: why3.File
    prior_attempts: Sequence[str]

    __parser__: ClassVar = dp.yaml_from_last_block
