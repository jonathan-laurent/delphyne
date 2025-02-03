"""
A baseline strategy that attempts at annotating the program from scratch until
Why3 accepts the proof. We write it in such a way to reuse the ProposeInvariants
query and the associated demonstrations.
"""

import delphyne as dp
from delphyne import Branch, Computation, Failure, Strategy, strategy

import why3_utils as why3
from find_invariants import ProposeInvariants


@strategy
def prove_program_baseline(
    prog: why3.File,
) -> Strategy[Branch | Failure | Computation, dp.PromptingPolicy, why3.File]:
    feedback = yield from dp.compute(why3.check, prog, prog)
    yield from dp.ensure(feedback.error is None, "invalid program")
    if feedback.success:
        return prog
    remaining = [o for o in feedback.obligations if not o.proved]
    unproved = remaining[0]
    yield from dp.ensure(len(remaining) == 1, "too many remaining obligations")
    invariants = yield from dp.branch(
        ProposeInvariants(unproved, [])(dp.PromptingPolicy, lambda p: p)
    )
    annotated = why3.add_invariants(prog, invariants)
    feedback = yield from dp.compute(why3.check, prog, annotated)
    yield from dp.ensure(feedback.success, "invalid invariants")
    return prog


def prove_program_baseline_policy(
    model_name: str = "gpt-4o",
) -> dp.Policy[Branch | Failure | Computation, dp.PromptingPolicy]:
    model = dp.openai_model(model_name)
    pp = dp.few_shot(model)
    return (dp.dfs() @ dp.elim_compute, pp)
