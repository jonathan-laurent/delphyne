"""
Using Delphyne and sympy to solve a simple symbolic math task.
"""

# pyright: strict
# fmt: off

from dataclasses import dataclass
from typing import assert_never

import sympy as sp  # type: ignore

import delphyne as dp
from delphyne import Branch, Fail, Strategy, strategy


@strategy
def find_param_value(
    expr: str,
) -> "Strategy[Branch | Fail, FindParamValueIP, int]":
    x, n = sp.Symbol("x", dummy=True, real=True), sp.Symbol("n", dummy=True)
    symbs = {"x": x, "n": n}
    try:
        n_val = yield from dp.branch(
            FindParamValue(expr)
                .using(lambda p: p.guess, FindParamValueIP))
        expr_sp = sp.parse_expr(expr, symbs).subs({n: n_val})
        equiv = yield from dp.branch(
            RewriteExpr(str(expr_sp))
                .using(lambda p: p.prove, FindParamValueIP))
        equiv_sp = sp.parse_expr(equiv, symbs)
        equivalent = (expr_sp - equiv_sp).simplify() == 0
        yield from dp.ensure(equivalent, "not_equivalent")
        yield from dp.ensure(equiv_sp.is_nonnegative, "not_nonneg")
        return n_val
    except Exception as e:
        assert_never((yield from dp.fail("sympy_error", message=str(e))))


@dataclass
class FindParamValue(dp.Query[int]):
    """
    Given a sympy expression featuring a real variable `x` and an
    integer parameter `n`, find an integer value for `n` such that the
    expression is non-negative for all real `x`. Terminate your answer
    with a code block delimited by triple backquotes containing an integer.
    """

    expr: str
    __parser__ = dp.last_code_block.yaml


@dataclass
class RewriteExpr(dp.Query[str]):
    """
    Given a sympy expression featuring variable `x`, rewrite it into an
    equivalent form that makes it clear that the expression is
    nonnegative for all real values of `x`. Terminate your answer with a
    code block delimited by triple backquotes. This block must contain a
    new sympy expression, or nothing if no rewriting could be found.
    """

    expr: str
    __parser__ = dp.last_code_block.trim


@dataclass
class FindParamValueIP:
    guess: dp.PromptingPolicy
    prove: dp.PromptingPolicy


@dp.ensure_compatible(find_param_value)
def serial_policy(
    model_name: dp.StandardModelName = "gpt-5-mini",
    proof_retries: int = 1
) -> dp.Policy[Branch | Fail, FindParamValueIP]:
    model = dp.standard_model(model_name)
    return dp.dfs() & FindParamValueIP(
        guess=dp.few_shot(model),
        prove=dp.take(proof_retries + 1) @ dp.few_shot(model))


@dp.ensure_compatible(find_param_value)
def parallel_policy(
    model_name: dp.StandardModelName = "gpt-5-mini",
    par_find: int = 2,
    par_rewrite: int = 2
) -> dp.Policy[Branch | Fail, FindParamValueIP]:
    model = dp.standard_model(model_name)
    return dp.loop() @ dp.par_dfs() & FindParamValueIP(
        guess=dp.few_shot(model, max_requests=1, num_completions=par_find),
        prove=dp.few_shot(model, max_requests=1, num_completions=par_rewrite))


if __name__ == "__main__":
    budget = dp.BudgetLimit({dp.NUM_REQUESTS: 2})
    res, _ = (
        find_param_value("2*x**2 - 4*x + n")
        .run_toplevel(dp.PolicyEnv(demonstration_files=[]), serial_policy())
        .collect(budget=budget, num_generated=1)
    )
    print(res[0].tracked.value)  # e.g. 2
