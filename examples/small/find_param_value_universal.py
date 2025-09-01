"""
Using Delphyne and sympy to solve a simple symbolic math task.

Version that uses universal queries.
"""

# pyright: strict

from typing import assert_never

import sympy as sp

import delphyne as dp
from delphyne import Branch, Fail, IPDict, Strategy, strategy


@strategy
def find_param_value(expr: str) -> Strategy[Branch | Fail, IPDict, int]:
    """
    Find an integer `n` that makes a given math expression nonnegative
    for all real `x`. Prove that the resulting expression is nonnegative
    by rewriting it into an equivalent form.
    """
    x, n = sp.Symbol("x", dummy=True, real=True), sp.Symbol("n", dummy=True)
    symbs = {"x": x, "n": n}
    try:
        n_val = yield from dp.guess(int, using=[expr])
        expr_sp = sp.parse_expr(expr, symbs).subs({n: n_val})
        equiv = yield from dp.guess(str, using=[str(expr_sp)])
        equiv_sp = sp.parse_expr(equiv, symbs)
        equivalent = (expr_sp - equiv_sp).simplify() == 0
        yield from dp.ensure(equivalent, "not_equivalent")
        yield from dp.ensure(equiv_sp.is_nonnegative, "not_nonneg")
        return n_val
    except Exception as e:
        assert_never((yield from dp.fail("sympy_error", message=str(e))))


def serial_policy():
    model = dp.standard_model("gpt-5-mini")
    return dp.dfs() & {
        "n_val": dp.few_shot(model),
        "equiv": dp.take(2) @ dp.few_shot(model),
    }


def parallel_policy():
    model = dp.standard_model("gpt-5-mini")
    return dp.loop() @ dp.par_dfs() & {
        "n_val": dp.few_shot(model, max_requests=1, num_completions=3),
        "equiv": dp.few_shot(model, max_requests=1, num_completions=2),
    }


if __name__ == "__main__":
    budget = dp.BudgetLimit({dp.NUM_REQUESTS: 2})
    res, _ = (
        find_param_value("2*x**2 - 4*x + n")
        .run_toplevel(dp.PolicyEnv(), serial_policy())
        .collect(budget=budget, num_generated=1)
    )
    print(res[0].tracked.value)  # e.g. 2
