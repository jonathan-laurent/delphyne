from dataclasses import dataclass
from typing import assert_never

import sympy as sp

import delphyne as dp
from delphyne import Branch, Fail, IPDict, Strategy, strategy


@strategy
def find_param_value(expr: str) -> Strategy[Branch | Fail, IPDict, int]:
    x, n = sp.Symbol("x", real=True), sp.Symbol("n", integer=True)
    symbs = {"x": x, "n": n}
    try:
        n_val = yield from dp.branch(FindParamValue(expr).using(...))
        expr_sp = sp.parse_expr(expr, symbs).subs({n: n_val})
        equiv = yield from dp.branch(RewriteExpr(str(expr_sp)).using(...))
        if equiv is None:
            assert_never((yield from dp.fail("no_rewrite")))
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
    with a triple-quoted code block containing an integer.
    """

    expr: str
    __parser__ = dp.yaml_from_last_block


@dataclass
class RewriteExpr(dp.Query[str | None]):
    """
    Given a sympy expression featuring variable `x`, rewrite it into an
    equivalent form that makes it clear that the expression is
    nonnegative for all real values of `x`. Terminate your answer with a
    triple-quoted code block containing either an expression as a string
    or `null` if the expression cannot be rewritten in such a way.
    """

    expr: str
    __parser__ = dp.yaml_from_last_block


def parse(expr: str) -> sp.Expr:
    x = sp.Symbol("x", real=True)
    n = sp.Symbol("n", integer=True)
    parsed = sp.parse_expr(expr, local_dict={"x": x, "n": n})
    assert parsed.free_symbols.issubset({x, n})
    return parsed
