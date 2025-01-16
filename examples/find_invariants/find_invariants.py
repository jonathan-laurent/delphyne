"""
A simple Delphyne strategy to discover loop invariants with Why3.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import delphyne as dp
from delphyne import Branch, Factor, Failure, Strategy, strategy

import why3_utils as why3


@dataclass
class ProveProgramIP:
    propose_change: "dp.Policy[Branch | Failure, ProposeChangeIP]"


@dataclass
class ProposeChangeIP:
    pass


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


@strategy
def prove_program(
    prog: why3.File,
) -> Strategy[Branch | Factor | Failure, ProveProgramIP, why3.File]:
    annotated: why3.File = prog
    while True:
        feedback = why3.check(prog, annotated)
        yield from dp.ensure(feedback.error is None, "invalid program")
        if feedback.success:
            return annotated
        # yield from dp.ensure()
        proposal = yield from dp.branch(
            dp.iterate(
                lambda prior: propose_change(prog, annotated, feedback, prior)(
                    ProveProgramIP, lambda p: p.propose_change)))  # fmt: skip
        annotated = proposal.annotated


@strategy
def propose_change(
    prog: why3.File,
    annotated: why3.File,
    feedback: why3.Feedback,
    prior_attempts: Sequence[Proposal] | None,
) -> Strategy[
    Branch | Failure,
    ProposeChangeIP,
    tuple[Proposal | None, Sequence[Proposal]],
]:
    assert False
    yield


@dataclass
class EvaluateProposal(dp.Query[float]):
    annotated: why3.File
