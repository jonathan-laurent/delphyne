"""
A simple Delphyne strategy to discover loop invariants with Why3.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import ClassVar

import delphyne as dp
from delphyne import Branch, Failure, Strategy, Value, strategy

import why3_utils as why3

#####
##### Policy Types
#####


@dataclass
class ProposeChangeIP:
    pass


@dataclass
class ProveProgramIP:
    propose_change: "dp.Policy[Branch | Failure, ProposeChangeIP]"
    eval_proposal: dp.PromptingPolicy
    quantify_eval: "Callable[[ProofStateMetrics], float]"


#####
##### Type Definitions
#####


@dataclass
class Proposal:
    """
    A proposal that was made to finish a proof.

    The proposal contains a full annotated program (`annotated`) but
    also a summary of the changes made, for easier comparison with
    previous proposals when doing redundancy checks. The
    `rejection_reason` field is not `None` when the proposal was
    rejected before consideration and contains a reason for the
    rejection, so that future attempts can try and avoid the same
    mistake.
    """

    change_summary: str
    annotated: why3.File
    rejection_reason: str | None


@dataclass
class ProofStateMetrics:
    num_annotations: int
    prob_provable: float


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
) -> Strategy[Branch | Value | Failure, ProveProgramIP, why3.File]:
    IP = ProveProgramIP  # shortcut for the inner policy type
    annotated: why3.File = prog
    while True:
        feedback = why3.check(prog, annotated)
        yield from dp.ensure(feedback.error is None, "invalid program")
        if feedback.success:
            return annotated
        yield from dp.value(
            EvaluateProofState(annotated)(IP, lambda p: p.eval_proposal),
            lambda p: p.quantify_eval)  # fmt:skip
        proposal = yield from dp.branch(
            dp.iterate(
                lambda prior: propose_change(annotated, feedback, prior)(
                    IP, lambda p: p.propose_change)))  # fmt: skip
        annotated = proposal.annotated


@strategy
def propose_change(
    annotated: why3.File,
    feedback: why3.Feedback,
    prior_attempts: Sequence[Proposal] | None,
) -> Strategy[
    Branch | Failure,
    ProposeChangeIP,
    tuple[Proposal | None, Sequence[Proposal]],
]:
    """
    Try and propose a better solution. We have two steps: first we
    propose. Then, we ask to summarize the change and check redundancy.
    """

    assert False
    yield


@dataclass
class EvaluateProofState(dp.Query[ProofStateMetrics]):
    annotated: why3.File

    def parse(self, mode: str | None, answer: str) -> ProofStateMetrics:
        metrics = dp.yaml_from_last_block(ProofStateMetrics, answer)
        prob_valid = 0 <= metrics.prob_provable <= 1
        num_annots_valid = metrics.num_annotations >= 1
        if not (prob_valid and num_annots_valid):
            raise dp.ParseError("Invalid metrics.")
        return metrics


@dataclass
class ProposeChange(dp.Query[why3.File]):
    annotated: why3.File

    __parser__: ClassVar = dp.string_from_last_block


@dataclass
class EvaluateProposal(dp.Query[ProposalEvaluation]):
    before: why3.File
    after: why3.File

    __parser__: ClassVar = dp.yaml_from_last_block


#####
##### Policy Definition
#####
