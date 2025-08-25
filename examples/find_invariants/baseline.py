"""
A conversational agent baseline implemented using `interact`, where an
agent repeatedly guesses invariants and is provided feedback until it
succeeds.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Never

import delphyne as dp
from delphyne import Branch, Compute, Strategy, dfs, strategy

import why3_utils as why3

# fmt: off


#####
##### Interactive Baseline
#####


@strategy
def prove_program_interactive(
    prog: why3.File,
) -> Strategy[Branch, dp.PromptingPolicy, why3.File]:
    annotated = yield from dp.interact(
        step=lambda prefix, _:
            AnnotateWithInvs(prog, prefix).using(dp.ambient_pp),
        process=lambda invs, _:
            check_invariants(prog, invs).using(dp.just_compute))
    return annotated


@dataclass
class AnnotateWithInvs(dp.Query[dp.Response[Sequence[why3.Formula], Never]]):
    prog: why3.File
    prefix: dp.AnswerPrefix

    __parser__ = dp.last_code_block.yaml.response


@strategy
def check_invariants(
    prog: why3.File, invariants: Sequence[why3.Formula]
) -> dp.Strategy[Compute, object, why3.File | dp.Error]:
    annotated = why3.add_invariants(prog, invariants)
    feedback = yield from dp.compute(why3.check, prog, annotated)
    if feedback.success:
        return annotated
    feedback.obligations = [
        o for o in feedback.obligations if not o.proved]
    return dp.Error(label="feedback", meta=feedback)


#####
##### Policy
#####


def prove_program_interactive_policy(
    model_name: dp.StandardModelName,
    temperature: float | None = None,
    max_feedback_cycles: int = 3,
    loop: bool = False,
):
    model = dp.standard_model(model_name)

    # `interact` branches twice per feedback cycle
    sp = dfs(max_depth=2*(max_feedback_cycles+1))
    if loop:
        sp = dp.loop() @ sp
    pp = dp.few_shot(model, temperature=temperature, max_requests=1)
    return sp & pp
