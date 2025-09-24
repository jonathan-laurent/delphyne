"""
A baseline strategy that attempts to annotate the program from scratch,
without feedback, until Why3 accepts the proof. We write it in such a
way to reuse the `ProposeInvariants` query from `abduct_and_branch` and
the associated demonstrations.
"""

import delphyne as dp
from delphyne import Branch, Compute, Fail, Strategy, strategy

import why3_utils as why3
from abduct_and_branch import ProposeInvariants


@strategy
def prove_program_one_guess(
    prog: why3.File,
) -> Strategy[Branch | Fail | Compute, dp.PromptingPolicy, why3.File]:
    feedback = yield from dp.compute(why3.check)(prog, prog)
    yield from dp.ensure(feedback.error is None, label="invalid_program")
    if feedback.success:
        return prog
    remaining = [o for o in feedback.obligations if not o.proved]
    unproved = remaining[0]
    yield from dp.ensure(len(remaining) == 1)
    invariants = yield from dp.branch(
        ProposeInvariants(unproved, []).using(lambda p: p, dp.PromptingPolicy),
    )
    annotated = why3.add_invariants(prog, invariants)
    feedback = yield from dp.compute(why3.check)(prog, annotated)
    yield from dp.ensure(feedback.success)
    return annotated


def prove_program_one_guess_policy(
    model_name: str = "gpt-4o",
) -> dp.Policy[Branch | Fail | Compute, dp.PromptingPolicy]:
    model = dp.openai_model(model_name)
    pp = dp.few_shot(model)
    return dp.dfs() @ dp.elim_compute() & pp
