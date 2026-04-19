"""
A Simple Delphyne Example

Problem: find an integer n that makes an expression nonnegative for all x.
Example: for x² - 2x + n, pick n = 1 since x² - 2x + 1 = (x - 1)² >= 0.

Strategy: to solve the problem reliably, first guess a value for n, and
then rewrite the resulting expression in such a way to prove that it is
clearly nonnegative.
"""

import sympy
import delphyne as dp


@dp.strategy
def find_param_value(expr: str):
    """
    Find an integer `n` that provably makes `expr` nonnegative.
    """
    x, n = sympy.Symbol("x", real=True), sympy.Symbol("n")
    symbs = {"x": x, "n": n}
    try:
        n_val = yield from dp.guess(int, using=[expr])
        expr_sp = sympy.parse_expr(expr, symbs).subs({n: n_val})
        equiv = yield from dp.guess(str, using=[str(expr_sp)])
        equiv_sp = sympy.parse_expr(equiv, symbs)
        equivalent = (expr_sp - equiv_sp).simplify() == 0
        yield from dp.ensure(equivalent, "not_equivalent")
        yield from dp.ensure(equiv_sp.is_nonnegative, "not_nonneg")
        return n_val
    except Exception as e:
        yield from dp.ensure(False, "sympy_error", message=str(e))


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