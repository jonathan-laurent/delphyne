from dataclasses import dataclass

import sympy as sp

#####
##### Proof Definition
#####

type StepId = int
type Term = str
type Eq = tuple[Term, Term]
type Step = tuple[Eq, Justification]
type Justification = Sym | Trans | Rule | PrevStep
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
    vars: dict[str, Term] | None = None


@dataclass
class PrevStep:
    step: StepId


#####
##### Key sympy functionality
#####


SYMPY_LOCALS = {"sin": sp.Function("sin"), "cos": sp.Function("cos")}
"""
We treat `sin` and `cos` as uninterpreted function symbols so that sympy
is not allowed to simplify them.
"""


def parse_eq(eq: Eq) -> tuple[sp.Expr, sp.Expr]:
    return parse_term(eq[0]), parse_term(eq[1])


def parse_term(term: Term) -> sp.Expr:
    return sp.sympify(term, locals=SYMPY_LOCALS)


def rewrite(term: Term, rule: Eq, vars: dict[str, Term]) -> Term:
    vars_parsed = {k: parse_term(v) for k, v in vars.items()}
    term_parsed = parse_term(term)
    lhs, rhs = parse_eq(rule)
    lhs = lhs.subs(vars_parsed)
    rhs = rhs.subs(vars_parsed)
    res = term_parsed.subs(lhs, rhs)
    return str(res)

def equal_terms(lhs: Term, rhs: Term) -> bool:
    return parse_term(lhs) == parse_term(rhs)


#####
##### Proof Definition
#####


@dataclass
class ProofError:
    feedback: str


def check_rule_application(eq: Eq, rule: Rule, rules: dict[str, Eq]) -> None:
    lhs, rhs = parse_eq(eq)
    rule_lhs, rule_rhs = parse_eq(rules[rule.rule])


def check(eq: Eq, proof: Proof, rules: dict[str, Eq]) -> tuple[bool, str]:
    pass