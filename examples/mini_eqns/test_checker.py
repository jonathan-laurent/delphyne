from checker import rewrite, equal_terms
import pytest


@pytest.mark.parametrize(
    "src, rule, vars, target",
    [
        ("sin(0) + 1", ("sin(0)", "0"), {}, "1"),
        ("exp(sin(-y))", ("sin(-x)", "sin(x)"), {"x": "y"}, "exp(sin(y))"),
        (
            "cos(a - b)",
            ("cos(x + y)", "cos(x)*cos(y) - sin(x)*sin(y)"),
            {"x": "a", "y": "-b"},
            "cos(a)*cos(-b) - sin(a)*sin(-b)",
        ),
    ],
)
def test_rewrite(src, rule, vars, target):
    rewritten = rewrite(src, rule, vars)
    assert equal_terms(rewritten, target)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ("1 + 2", "3"),
        ("sqrt(4)", "2"),
        ("pi/2 - pi/3", "pi/6"),
    ],
)
def test_equal_terms(lhs, rhs):
    assert equal_terms(lhs, rhs)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ("1 + 2", "4"),
        # `sin` and `cos` are treated as uninterpreted function symbols
        ("cos(0)", "1"),
    ],
)
def test_not_equal_terms(lhs, rhs):
    assert not equal_terms(lhs, rhs)
