from dataclasses import dataclass, field
from typing import Any

import pydantic
import sympy as sp  # type: ignore
import yaml

#####
##### Proof Definition
#####

type StepId = int
type Term = str
type Eq = tuple[Term, Term]
type Step = tuple[Eq, Justification]
type Justification = Sym | Trans | Rule | RewritePrev
type Proof = dict[StepId, Step]


@dataclass
class Trans:
    trans: list[StepId]


@dataclass
class Sym:
    sym: StepId


@dataclass
class Rule:
    rule: str
    vars: dict[str, Term] = field(default_factory=dict[str, Term])


@dataclass
class RewritePrev:
    step: StepId


@dataclass
class ParseError:
    msg: str


def parse_proof(proof_str: str) -> Proof | ParseError:
    try:
        loaded = yaml.safe_load(proof_str)
        return pydantic.TypeAdapter(Proof).validate_python(loaded)  # type: ignore
    except Exception as e:
        return ParseError(str(e))


#####
##### Key sympy functionality
#####


SYMPY_LOCALS: dict[str, Any] = {
    "sin": sp.Function("sin"),
    "cos": sp.Function("cos"),
}
"""
We treat `sin` and `cos` as uninterpreted function symbols so that sympy
is not allowed to simplify them.
"""


def parse_eq(eq: Eq) -> tuple[sp.Expr, sp.Expr]:
    return parse_term(eq[0]), parse_term(eq[1])


def parse_term(term: Term) -> sp.Expr:
    return sp.sympify(term, locals=SYMPY_LOCALS)  # type: ignore


def rewrite(term: Term, rule: Eq, vars: dict[str, Term]) -> Term:
    vars_parsed = {k: parse_term(v) for k, v in vars.items()}
    term_parsed = parse_term(term)
    lhs, rhs = parse_eq(rule)
    lhs = lhs.subs(vars_parsed)  # type: ignore
    rhs = rhs.subs(vars_parsed)  # type: ignore
    res = term_parsed.subs(lhs, rhs)  # type: ignore
    return str(res)  # type: ignore


def equal_terms(lhs: Term, rhs: Term) -> bool:
    lhs_e, rhs_e = parse_term(lhs), parse_term(rhs)
    return lhs_e == rhs_e or (sp.expand(lhs_e - rhs_e) == 0)  # type: ignore


#####
##### Proof Definition
#####


@dataclass
class ProofError(Exception):
    feedback: str
    step: int | None = None

    def __str__(self):
        if self.step is not None:
            return f"Error at step {self.step}: {self.feedback}"
        return self.feedback


def check_rule_application(eq: Eq, rule_eq: Eq, vars: dict[str, Term]) -> None:
    lhs, rhs = eq
    new_lhs = rewrite(lhs, rule_eq, vars)
    if not equal_terms(new_lhs, rhs):
        raise ProofError(
            f"Rule application failed. Obtained: '{new_lhs}' instead of "
            + "the expected right-hand side.",
        )


def check(eq: Eq, proof: Proof, rules: dict[str, Eq]) -> ProofError | None:
    def ensure_valid_prev_step(step_id: StepId, *, cur_step: StepId):
        if step_id not in proof:
            raise ProofError(f"Step {step_id} not found")
        if step_id >= cur_step:
            raise ProofError(f"Step {step_id} is not a past step.")

    def check_equal_terms(lhs: Term, rhs: Term):
        if not equal_terms(lhs, rhs):
            raise ProofError(f"Unequal terms: '{lhs}' and '{rhs}'")

    def step_eq(step_id: StepId) -> Eq:
        return proof[step_id][0]

    for cur_step, (cur_eq, justification) in proof.items():
        try:
            match justification:
                case Sym(step):
                    ensure_valid_prev_step(step, cur_step=cur_step)
                    valid_1 = equal_terms(cur_eq[0], step_eq(step)[1])
                    valid_2 = equal_terms(cur_eq[1], step_eq(step)[0])
                    if not (valid_1 and valid_2):
                        raise ProofError("Application of symmetry failed.")
                case Trans(steps):
                    for step in steps:
                        ensure_valid_prev_step(step, cur_step=cur_step)
                    n = len(steps)
                    if n < 2:
                        msg = "Transitivity requires at least 2 steps."
                        raise ProofError(msg)
                    check_equal_terms(cur_eq[0], step_eq(steps[0])[0])
                    for i in range(1, n):
                        check_equal_terms(
                            step_eq(steps[i - 1])[1], step_eq(steps[i])[0]
                        )
                    check_equal_terms(cur_eq[1], step_eq(steps[-1])[1])
                case Rule(rule, vars):
                    if rule not in rules:
                        raise ProofError(f"Rule '{rule}' not found")
                    rule_eq = rules[rule]
                    check_rule_application(cur_eq, rule_eq, vars)
                case RewritePrev(prev_step_id):
                    ensure_valid_prev_step(prev_step_id, cur_step=cur_step)
                    rule_eq = step_eq(prev_step_id)
                    check_rule_application(cur_eq, rule_eq, {})
        except ProofError as e:
            e.step = cur_step
            return e
    if proof:
        last_eq = proof[max(proof.keys())][0]
        if equal_terms(last_eq[0], eq[0]) and equal_terms(last_eq[1], eq[1]):
            return None
    return ProofError("The proof does not end with the equation to be proved.")


TRIG_RULES = {
    "cos_zero": ("cos(0)", "1"),
    "sin_zero": ("sin(0)", "0"),
    "sin_halfpi": ("sin(pi/2)", "1"),
    "cos_halfpi": ("cos(pi/2)", "0"),
    "sin_neg": ("sin(-x)", "-sin(x)"),
    "cos_neg": ("cos(-x)", "cos(x)"),
    "cos_add": ("cos(x + y)", "cos(x)*cos(y) - sin(x)*sin(y)"),
    "sin_add": ("sin(x + y)", "sin(x)*cos(y) + cos(x)*sin(y)"),
}
