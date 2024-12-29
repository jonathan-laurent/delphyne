"""
Three strategies for computing invariants
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Literal

import why3py.simple as why3

from delphyne import stdlib as std
from delphyne.stdlib import Branching, Params, dfs, iterated, strategy
from delphyne.stdlib.caching import cache
from delphyne.stdlib.nodes import branch, ensure, fail
from delphyne.stdlib.search.bfs import BFS, bfs, bfs_branch, bfs_factor


#####
##### Why3 utilities and caching
#####


type Formula = str


@dataclass
class Why3Input:
    prog: why3.File
    original: why3.File | None


@cache(Path(__file__).parent / "cache" / "why3")
def check_why3_file(input: Why3Input) -> why3.Outcome:
    return why3.check_file(input.prog, input.original)


@dataclass
class LightObligation:
    name: str

    annotated: str
    context: str
    goal: str

    @staticmethod
    def from_why3(obl: why3.Obligation) -> "LightObligation":
        return LightObligation(
            name=obl.name,
            annotated=obl.annotated,
            context=obl.context,
            # Right now, the feedback is nondeterministic because
            # `obl.goal` has the form "goal ...vc<i>: ..." where i is an
            # internal counter. Thus, we get rid of this part.
            goal=extract_goal_formula(obl.goal),
        )


@dataclass
class LightFeedback:
    error: str | None
    obligations: list[LightObligation]

    @staticmethod
    def from_why3(outcome: why3.Outcome) -> "LightFeedback":
        if outcome.kind == "error" or outcome.kind == "mismatch":
            return LightFeedback(error=why3.summary(outcome), obligations=[])
        return LightFeedback(
            error=None,
            obligations=[
                LightObligation.from_why3(obl)
                for obl in outcome.obligations
                if not obl.proved
            ],
        )


def add_invariant(prog: why3.File, inv: Formula) -> why3.File:
    # TODO: make it more robust
    lines = prog.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("  while"):
            lines.insert(i + 1, f"    invariant {{ {inv} }}")
            return "\n".join(lines)
    assert False


def check_proposal(prog: why3.File, original: why3.File | None) -> bool:
    res = check_why3_file(Why3Input(prog, original))
    feedback = LightFeedback.from_why3(res)
    return (
        isinstance(res, why3.Obligations)
        and len(feedback.obligations) <= MAX_OPEN_OBLIGATIONS
    )


def extract_goal_formula(goal: str) -> str:
    """
    From a string of the form "goal ...: <fml>", extract fml.
    """
    start = goal.index(":") + 1
    return goal[start:].strip()


#####
##### Single prompt version
#####


@dataclass
class ProposeAnnotations(std.StructuredQuery[Params, why3.File]):
    prog: why3.File

    def parser(self) -> std.Parser:
        return std.string_from_last_block


@strategy(dfs)
def annotate_invariants_with_single_prompt(
    prog: why3.File,
) -> Branching[Params, why3.File]:
    annotated = yield from branch(ProposeAnnotations(prog))
    res = check_why3_file(Why3Input(annotated, original=prog))
    if not (res.kind == "obligations" and res.success):
        yield from fail(why3.summary(res))
    return annotated


#####
##### Propose and repair
#####


@dataclass
class ProposeAndRepairParams(std.HasMaxDepth, Params):
    max_depth: int = 3
    repair_branching: int = 2

    def get_max_depth(self) -> int:
        return self.max_depth


@strategy(dfs)
def annotate_invariants_with_propose_and_repair(
    prog: why3.File,
) -> Branching[ProposeAndRepairParams, why3.File]:
    annotated = yield from branch(ProposeAnnotations(prog))
    while True:
        res = check_why3_file(Why3Input(annotated, original=prog))
        if res.kind == "obligations" and res.success:
            return annotated
        feedback = LightFeedback.from_why3(res)
        if feedback.error:
            yield from fail(feedback.error)
        annotated = yield from branch(
            FixAnnotations(prog, annotated, feedback),
            max_branching=(lambda p: p.repair_branching),
            param_type=ProposeAndRepairParams,
        )


@dataclass
class FixAnnotations(std.StructuredQuery[Params, why3.File]):
    prog: why3.File
    annotated: why3.File
    feedback: LightFeedback

    def parser(self) -> std.Parser:
        return std.string_from_last_block


#####
##### Incremental invariant discovery
#####


MAX_INVARIANTS = 3
MAX_OPEN_OBLIGATIONS = 1


@dataclass
class ProposeInvariant(std.StructuredQuery[Params, Formula]):
    prog: why3.File
    feedback: LightFeedback
    proposed_already: list[Formula]

    def parser(self) -> std.Parser:
        return std.trimmed_string_from_last_block


type ConfidenceScore = Literal[1, 2, 3, 4, 5]


@dataclass
class EvaluateInvariantProposal(std.StructuredQuery[Params, ConfidenceScore]):
    prog: why3.File
    inv: Formula

    def parser(self) -> std.Parser:
        return std.yaml_from_last_block


@dataclass
class InvSearchParams(Params):
    proposal_priors: Sequence[float] = (1, 0.6, 0.2, 0)
    max_proposal_attempts: int = 4


@strategy(dfs)
def propose_invariant(
    prog: why3.File,
    feedback: LightFeedback,
    blacklist: list[Formula],
) -> Branching[InvSearchParams, Formula]:
    proposal = yield from branch(
        ProposeInvariant(prog, feedback, blacklist),
        max_branching=lambda p: p.max_proposal_attempts,
        param_type=InvSearchParams,
    )
    new_prog = add_invariant(prog, proposal)
    is_constructive = check_proposal(new_prog, prog)
    yield from ensure(is_constructive, "Proposal did not pass sanity checks.")
    return proposal


@strategy(dfs)
def evaluate_proposal(
    prog: why3.File,
    inv: Formula,
) -> Branching[Params, float]:
    score = yield from branch(EvaluateInvariantProposal(prog, inv))
    return score / 5


@strategy(bfs)
def annotate_invariants_incrementally(
    prog: why3.File,
) -> BFS[InvSearchParams, why3.File]:
    annotated = prog
    i = 0
    while True:
        i += 1
        res = check_why3_file(Why3Input(annotated, original=prog))
        if not isinstance(res, why3.Obligations):
            yield from fail(why3.summary(res))
            assert False
        if res.success:
            return annotated
        if i > MAX_INVARIANTS:
            yield from fail("Too many invariants were proposed.")
        feedback = LightFeedback.from_why3(res)
        proposed = yield from bfs_branch(
            iterated(partial(propose_invariant, annotated, feedback)),
            confidence_priors=lambda p: p.proposal_priors,
            param_type=InvSearchParams,
            label=f"proposal_{i}",
        )
        yield from bfs_factor(evaluate_proposal(annotated, proposed))
        annotated = add_invariant(annotated, proposed)
