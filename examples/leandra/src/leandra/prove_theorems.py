"""
Main Strategy for Proving Lean Theorems.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Never, assert_never

import delphyne as dp
import lean_interact.interface as li_intf
from delphyne import Branch, Compute, Fail, Strategy, strategy

import leandra.find_theorems as lft
from leandra.dsl import LeanProof, LeanTheorem, ProofSketch, compile_sketch
from leandra.find_theorems import TheoremRequest, find_theorem
from leandra.tools import run_lean_command

# fmt: off


#####
#####  Top-Level Strategy
#####


@strategy
def prove_theorem(
    theorem: LeanTheorem,
) -> Strategy["_ProveTheoremSig", "ProveTheoremIP", LeanProof]:
    """
    Top-level Strategy for proving a Lean theorem.

    This strategy first decides based on `ProofTechniqueFlag` whether to
    directly ask an LLM for a proof or to ask for a sketched to be
    filled in (default).

    In the latter case, we first ask for a sketch using `sketch_proof`
    and then join calls to `fill_hole` for each hole in the sketch.
    """

    if (yield from dp.get_flag(ProofTechniqueFlag)) == "direct":
        sketch, goals = yield from direct_proof(theorem)
    else:
        sketch, goals = yield from dp.branch(
            sketch_proof(theorem).using(lambda p: p.sketch, ProveTheoremIP))
    # Even an empty sketch results in one goal.
    assert goals
    subproofs = yield from dp.join(
        [fill_hole(theorem, sketch, i, goal) for i, goal in enumerate(goals)])
    full_proof = compile_sketch(theorem, sketch, subproofs)
    response = yield from dp.compute(run_lean_command)(full_proof)
    # The `fill_hole` strategy guarantees that all holes are filled correctly.
    assert len(response.sorries) == 0
    # It is possible in theory that although each subproof is correct,
    # the full proof is not accepted (naming clashes, improper use of
    # metavariables...). This should be very rare though.
    if _has_errors(response):
        meta = (_lean_error_metadata(full_proof, response),)
        err = dp.Error(label="stitching_failure", meta=meta)
        assert_never((yield from dp.fail(error=err)))
    return full_proof


type _ProveTheoremSig = (
    Branch | Fail | Compute | dp.Join | dp.Flag[ProofTechniqueFlag])
"""
Signature for the `prove_theorem` strategy.
"""


@dataclass
class ProofTechniqueFlag(dp.FlagQuery[Literal["sketch", "direct"]]):
    """
    Flag that indicates whether to produce a sketch (default) or not.
    """


type Goals = Sequence[str]
"""
A goal corresponding to a hole in a proof sketch.
"""


def direct_proof(
    theorem: LeanTheorem,
) -> Strategy[Compute, "ProveTheoremIP", tuple[ProofSketch, Goals]]:
    """
    Produce a direct proof of the theorem.
    
    !!! note
        This strategy is meant to be inlined, hence the absence of the
        `@strategy` decorator.
    """
    sketch = ProofSketch(steps=(), comment=None)
    compiled = compile_sketch(theorem, sketch, [None])
    response = yield from dp.compute(run_lean_command)(compiled)
    assert not _has_errors(response)
    assert len(response.sorries) == 1
    return sketch, [response.sorries[0].goal]


@strategy
def fill_hole(
    theorem: LeanTheorem, sketch: ProofSketch, hole_index: int, goal: str
) -> Strategy[Branch, "ProveTheoremIP", LeanProof]:
    """
    Fill a specific hole via the `prove_subgoal` strategy.
    """
    proof = yield from dp.branch(
        prove_subgoal(theorem, sketch, hole_index, goal)
            .using(lambda p: p.subgoal, ProveTheoremIP))
    return proof


#####
##### Sketching Strategy
#####


@strategy
def sketch_proof(
    theorem: LeanTheorem,
) -> Strategy[Branch, "SketchProofIP", tuple[ProofSketch, Goals]]:
    """
    Interactively try to find a proof sketch.

    Whenever the sketch is invalid, the LLM is provided with the
    associated error messages.
    """
    IP = SketchProofIP
    sketch_and_goals = yield from dp.interact(
        step=lambda prefix, _:
            SketchProof(theorem, prefix).using(lambda p: p.step, IP),
        process=lambda sketch, _:
            check_sketch(theorem, sketch).using(lambda p: p.check, IP))
    return sketch_and_goals


@dataclass
class SketchProof(dp.Query[dp.Response[ProofSketch, Never]]):
    """
    Main query underlying the `sketch_proof` strategy.

    Feedback messages with label `sketch_has_errors` are possible.
    """

    theorem: LeanTheorem
    prefix: dp.AnswerPrefix
    __parser__ = dp.structured.response


@strategy
def check_sketch(
    theorem: LeanTheorem, sketch: ProofSketch
) -> Strategy[Compute, None, tuple[ProofSketch, Goals] | dp.Error]:
    """
    Check that a sketch is valid.

    A sketch is valid if it results in no errors (although it may still
    contain sorries).
    """
    compiled = compile_sketch(theorem, sketch, [None] * sketch.num_holes())
    response = yield from dp.compute(run_lean_command)(compiled)
    if _has_errors(response):
        return dp.Error(
            label="sketch_has_errors",
            meta=_lean_error_metadata(compiled, response),
        )
    # If both our sketch compiler and the Lean REPL are correct, there
    # should be as many remaining sorries as goals in the sketch.
    assert len(response.sorries) == sketch.num_holes()
    goals = [s.goal for s in response.sorries]
    return sketch, goals


def _lean_error_metadata(
    lean_command: str, response: li_intf.BaseREPLResponse
) -> dict[str, Any]:
    """
    Produce feedback metadata for when a Lean command fails.
    """
    errors = [m for m in response.messages if m.severity == "error"]
    warnings = [m for m in response.messages if m.severity == "warning"]
    return {
        "command": _annotate_with_line_numbers(lean_command),
        "errors": [_render_lean_message(m, line_info=True) for m in errors],
        "warnings": [_render_lean_message(m, line_info=True) for m in warnings],
    }


def _render_lean_message(message: li_intf.Message, *, line_info: bool) -> str:
    """
    Render a Lean message as a string, possibly including location info.
    """
    if line_info:
        start_line = message.start_pos.line
        if message.end_pos is not None and message.end_pos.line != start_line:
            end_line = message.end_pos.line
            line_header = f"lines {start_line}-{end_line}"
        else:
            line_header = f"line {start_line}"
        header = f"[{line_header}, {message.severity}]"
    else:
        header = f"[{message.severity}]"
    return f"[{header}] {message.data}"


def _annotate_with_line_numbers(lean_code: str) -> str:
    lines = lean_code.splitlines()
    return "\n".join(f"{line}  # line {i}" for i, line in enumerate(lines))


def _has_errors(response: li_intf.BaseREPLResponse) -> bool:
    return not any(m.severity == "error" for m in response.messages)


#####
#####  Hole-Filling Strategy
#####


@strategy
def prove_subgoal(
    theorem: LeanTheorem, sketch: ProofSketch, hole_index: int, goal: str
) -> Strategy[Branch, "ProveSubgoalIP", LeanProof]:
    """
    Prove a subgoal corresponding to a goal in the sketch.

    The full surrounding sketch is used as context, not for LLMs but for
    verifying the proof. Doing so would not be necessary if there was an
    easy way in Lean to parse a goal directly.

    This strategy proceeds interactively. The LLM is provided with a
    tool for finding theorems (implemented via the `find_theorem`
    strategy). Feedback is provided when incorrect proofs are proposed.
    """
    
    @strategy
    def process_proof(
        proof: LeanProof
    ) -> Strategy[Compute, None, LeanProof | dp.Error]:
        # Note: Delphyne has no problem with defining nested strategies.
        proofs: list[str | None] = [None] * sketch.num_holes()
        proofs[hole_index] = proof
        compiled = compile_sketch(theorem, sketch, proofs)
        response = yield from dp.compute(run_lean_command)(compiled)
        if response.sorries:
            return dp.Error(label="sub_proof_with_sorry")
        if _has_errors(response):
            return dp.Error(
                label="sub_proof_has_errors",
                meta=_lean_error_metadata_without_line_info(response))
        return proof

    proof = yield from dp.interact(
        step=lambda prefix, _:
            ProveGoal(goal, prefix).using(lambda p: p.step, ProveSubgoalIP),
        process=lambda proof, _:
            process_proof(proof).using(lambda p: p.check, ProveSubgoalIP),
        tools={TheoremRequest: lambda call:
            # In case `find_theorem` fails, we should not fail the
            # entire search and just report a tool call failure.
            dp.nofail(
                find_theorem(call).using(lambda p: p.find_thm, ProveSubgoalIP),
                default=None)})
    return proof


@dataclass
class ProveGoal(dp.Query[dp.Response[LeanProof, TheoremRequest]]):
    """
    Main query underlying the `prove_subgoal` strategy.

    Feedback messages with labels `sub_proof_has_errors` and
    `sub_proof_with_sorry` must be expected.

    Calls to the `TheoremRequest` tool can result in `None` in case of
    tool failure, which must be made clear in prompts.
    """
    goal: str
    prefix: dp.AnswerPrefix
    __parser__ = dp.get_text.trim.response


def _lean_error_metadata_without_line_info(
    response: li_intf.BaseREPLResponse
) -> dict[str, Any]:
    """
    Produce feedback metadata without location information.

    !!! note
        The current implementation makes it nontrivial to provide
        location information when a sub-proof fails, since what is sent
        to Lean includes the full surrounding sketch as context. This
        may improve in the future. 
    """
    errors = [m for m in response.messages if m.severity == "error"]
    warns = [m for m in response.messages if m.severity == "warning"]
    return {
        "errors": [_render_lean_message(m, line_info=False) for m in errors],
        "warnings": [_render_lean_message(m, line_info=False) for m in warns],
    }


#####
##### Internal Policy Types
#####

# fmt: on


@dataclass
class ProveTheoremIP:
    sketch: dp.Policy[Branch, "SketchProofIP"]
    subgoal: dp.Policy[Branch, "ProveSubgoalIP"]


@dataclass
class SketchProofIP:
    step: dp.PromptingPolicy
    check: dp.Policy[Compute, None]


@dataclass
class ProveSubgoalIP:
    step: dp.PromptingPolicy
    check: dp.Policy[Compute, None]
    find_thm: dp.Policy[Compute | Branch, lft.FindTheoremIP]


#####
##### Policies
#####


@dataclass
class ProveTheoremPolicy(dp.PolicyRecord[_ProveTheoremSig, ProveTheoremIP]):
    """
    Standard policy for `prove_theorem`.

    Attributes:
        find_theorem: Policy for finding theorems.
        lean_timeout: Timeout in seconds for checking sketches or proofs
            in direct modes.
    """

    find_theorem: lft.FindTheoremPolicy
    lean_timeout: float = 8.0

    def instantiate(self):
        lean_compute_args = {"timeout_in_seconds": self.lean_timeout}
        elim_lean_compute = dp.elim_compute(override_args=lean_compute_args)

        ip = ProveTheoremIP()  # type: ignore
        sp = (
            dp.dfs()
            @ dp.elim_join()
            @ dp.elim_flag(ProofTechniqueFlag, "sketch")
            @ elim_lean_compute
        )
        return sp & ip
