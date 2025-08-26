"""
Using Delphyne and sympy to solve a simple symbolic math task.
"""

# pyright: strict
# fmt: off

from dataclasses import dataclass
from typing import assert_never

import sympy as sp

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
            FindParamValue(expr).using(lambda p: p.find, FindParamValueIP))
        expr_sp = sp.parse_expr(expr, symbs).subs({n: n_val})
        equiv = yield from dp.branch(
            RewriteExpr(str(expr_sp))
            .using(lambda p: p.rewrite, FindParamValueIP))
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
    find: dp.PromptingPolicy
    rewrite: dp.PromptingPolicy


@dp.ensure_compatible(find_param_value)
def serial_policy(model_name: dp.StandardModelName = "gpt-5-mini"):
    model = dp.standard_model(model_name)
    return dp.dfs() & FindParamValueIP(
        find=dp.few_shot(model),
        rewrite=dp.take(2) @ dp.few_shot(model))


@dp.ensure_compatible(find_param_value)
def parallel_policy(model_name: dp.StandardModelName = "gpt-5-mini"):
    model = dp.standard_model(model_name)
    return dp.loop() @ dp.par_dfs() & FindParamValueIP(
        find=dp.take(3) @ dp.few_shot(model),
        rewrite=dp.take(2) @ dp.few_shot(model))
