import pathlib

import pytest

import checker as ch
from checker import Eq, Term, equal_terms, rewrite


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
def test_rewrite(src: Term, rule: Eq, vars: dict[str, Term], target: Term):
    rewritten = rewrite(src, rule, vars)
    assert equal_terms(rewritten, target)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ("1 + 2", "3"),
        ("sqrt(4)", "2"),
        ("pi/2 - pi/3", "pi/6"),
        (
            "-sin(-x)*sin(x) + cos(-x)*cos(x)",
            "cos(x)*cos(-x) - sin(x)*sin(-x)",
        ),
    ],
)
def test_equal_terms(lhs: Term, rhs: Term):
    assert equal_terms(lhs, rhs)


@pytest.mark.parametrize(
    "lhs, rhs",
    [
        ("1 + 2", "4"),
        # `sin` and `cos` are treated as uninterpreted function symbols
        ("cos(0)", "1"),
    ],
)
def test_not_equal_terms(lhs: Term, rhs: Term):
    assert not equal_terms(lhs, rhs)


def load_proof(name: str) -> ch.Proof | ch.ParseError:
    name = name + ".yaml"
    with open(pathlib.Path(__file__).parent / "proofs" / name) as f:
        proof_str = f.read()
        return ch.parse_proof(proof_str)


def test_proof_cos():
    proof = load_proof("cos_sin_sq")
    assert not isinstance(proof, ch.ParseError)
    error = ch.check(("cos(x)**2 + sin(x)**2", "1"), proof, ch.TRIG_RULES)
    assert error is None
