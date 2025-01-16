"""
A simple Delphyne strategy to discover loop invariants with Why3.
"""

from dataclasses import dataclass

import delphyne as dp
from delphyne import Branch, Factor, Failure, Strategy, strategy

import why3_utils as why3


@dataclass
class FindInvariantsIP:
    pass


@dataclass
class Proposal:
    change_summary: str
    annotated: why3.File
    success: bool


@strategy
def find_invariants(
    prog: why3.File,
) -> Strategy[Branch | Factor | Failure, FindInvariantsIP, why3.File]:
    annotated = prog
    feedback = why3.check(prog, annotated)
    yield from dp.ensure(feedback.error is None, "invalid program")
    while True:
        # We treat it like a proposal
        proposal = yield from dp.branch(propose_change(prog, annotated))
        pass


# @strategy
# def propose_change(prog: why3.File, annotated: why3.File, prior_attempts) -> Strategy[]:
#     pass
