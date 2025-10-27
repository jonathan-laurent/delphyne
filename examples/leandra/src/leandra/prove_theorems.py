"""
Main Strategy for Proving Lean Theorems
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Never, assert_never

import delphyne as dp
import lean_interact.interface as li_intf
from delphyne import Branch, Compute, Fail, Strategy, strategy

# import leandra.loogle_utils as loogle
from leandra.dsl import LeanProof, LeanTheorem, ProofSketch, compile_sketch

# from leandra.find_theorems import TheoremRequest, find_theorem
from leandra.tools import run_lean_command

#####
#####  Top-Level Strategy
#####


@strategy
def prove_theorem(
    theorem: LeanTheorem,
) -> Strategy[
    Branch | Fail | Compute | dp.Flag["ProofTechniqueFlag"] | dp.Join,
    "ProveTheoremIP",
    LeanProof,
]:
    if (yield from dp.get_flag(ProofTechniqueFlag)) == "direct":
        # ret = yield from dp.compute(run_lean_command)(theorem)
        assert False
    sketch, goals = yield from dp.branch(
        sketch_proof(theorem).using(lambda p: p.sketch, ProveTheoremIP)
    )
    assert goals
    subproofs = yield from dp.join(
        [fill_hole(theorem, sketch, i, goal) for i, goal in enumerate(goals)]
    )
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


type Goals = Sequence[str]


@dataclass
class ProofTechniqueFlag(dp.FlagQuery[Literal["sketch", "direct"]]):
    pass


#####
##### Sketching Strategy
#####


@strategy
def sketch_proof(
    theorem: LeanTheorem,
) -> Strategy[Branch, "SketchProofIP", tuple[ProofSketch, Goals]]:
    """
    Interactively try to find a proof sketch.
    """
    IP = SketchProofIP
    # fmt: off
    sketch_and_goals = yield from dp.interact(
        step=lambda prefix, _:
            SketchProof(theorem, prefix).using(lambda p: p.step, IP),
        process=lambda sketch, _:
            check_sketch(theorem, sketch).using(lambda p: p.check, IP))
    # fmt: on
    return sketch_and_goals


@dataclass
class SketchProof(dp.Query[dp.Response[ProofSketch, Never]]):
    theorem: LeanTheorem
    prefix: dp.AnswerPrefix


@strategy
def check_sketch(
    theorem: LeanTheorem, sketch: ProofSketch
) -> Strategy[Compute, None, tuple[ProofSketch, Goals] | dp.Error]:
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
    errors = [m for m in response.messages if m.severity == "error"]
    warnings = [m for m in response.messages if m.severity == "warning"]
    return {
        "command": _annotate_with_line_numbers(lean_command),
        "errors": [_render_lean_message(m) for m in errors],
        "warnings": [_render_lean_message(m) for m in warnings],
    }


def _render_lean_message(message: li_intf.Message) -> str:
    start_line = message.start_pos.line
    if message.end_pos is not None and message.end_pos.line != start_line:
        end_line = message.end_pos.line
        line_info = f"lines {start_line}-{end_line}"
    else:
        line_info = f"line {start_line}"
    return f"[{line_info}, {message.severity}] {message.data}"


def _annotate_with_line_numbers(lean_code: str) -> str:
    lines = lean_code.splitlines()
    return "\n".join(f"{line}  # line {i}" for i, line in enumerate(lines))


def _has_errors(response: li_intf.BaseREPLResponse) -> bool:
    return not any(m.severity == "error" for m in response.messages)


#####
#####  Hole-Filling Strategy
#####


@strategy
def fill_hole(
    theorem: LeanTheorem, sketch: ProofSketch, hole_index: int, goal: str
) -> Strategy[Branch, "ProveTheoremIP", LeanProof]:
    assert False
    yield


# tools = {TheoremRequest: lambda call:
#     find_theorem(call.request).using(lambda p: p.find_theorem, IP)
#          }


#####
##### Policies
#####


@dataclass
class ProveTheoremIP:
    sketch: dp.Policy[Branch, "SketchProofIP"]


@dataclass
class SketchProofIP:
    step: dp.PromptingPolicy
    check: dp.Policy[Compute, None]
