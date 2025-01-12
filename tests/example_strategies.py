"""
Simple example strategies.

The strategies defined in this file are used to test `Tree` but also to
test the server (see `test_server`).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Never, Sequence, TypeAlias, cast

import delphyne as dp

#####
##### MakeSum
#####


@dataclass
class MakeSum(dp.Query[list[int]]):
    allowed: list[int]
    goal: int
    __parser__: ClassVar = dp.raw_yaml


@dataclass
class MakeSumIP:
    make_sum: dp.PromptingPolicy


@dp.strategy
def make_sum(
    allowed: list[int], goal: int
) -> dp.Strategy[dp.Branch | dp.Failure, MakeSumIP, list[int]]:
    xs = yield from dp.branch(
        cands=MakeSum(allowed, goal).using(lambda p: p.make_sum),
        inner_policy_type=MakeSumIP,
    )
    yield from dp.ensure(all(x in allowed for x in xs), "forbidden-num")
    yield from dp.ensure(sum(xs) == goal, "wrong-sum")
    return xs


#####
##### Adding support for conjecture nodes
#####


@dataclass(frozen=True)
class Conjecture(dp.Node):
    cands: dp.OpaqueSpace[Any, Any]
    disprove: Callable[[dp.Tracked[Any]], dp.OpaqueSpace[Any, None]]
    aggregate: Callable[
        [tuple[dp.Tracked[Any], ...]], dp.OpaqueSpace[Any, Sequence[Any]]
    ]

    def navigate(self) -> dp.Navigation:
        return (yield self.cands)

    def primary_space(self):
        return self.cands


def make_conjecture[P, T](
    cands: dp.Builder[dp.OpaqueSpace[P, T]],
    disprove: Callable[[T], dp.Builder[dp.OpaqueSpace[P, None]]],
    aggregate: Callable[
        [tuple[T, ...]], dp.Builder[dp.OpaqueSpace[P, Sequence[T]]]
    ],
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Conjecture, P, T]:
    cand = yield dp.spawn_node(
        Conjecture, cands=cands, disprove=disprove, aggregate=aggregate
    )
    return cast(T, cand)


@dp.search_policy
async def just_guess[P, T](
    tree: dp.Tree[Conjecture, P, T], env: dp.PolicyEnv, policy: P
) -> dp.Stream[T]:
    """
    Do a DFS, treating conjecture nodes as simple branching nodes.
    """
    match tree.node:
        case dp.Success(x):
            yield dp.Yield(x)
        case Conjecture(candidate):
            async for msg in dp.bind_stream(
                candidate.stream(env, policy),
                lambda y: just_guess()(tree.child(y), env, policy),
            ):
                yield msg


#####
##### Utility: single prompting policy
#####


def one_pp(p: dp.PromptingPolicy) -> dp.PromptingPolicy:
    return p


#####
##### A more complex strategy with counterexamples
#####


# Pydantic does not work with Python 3.12 `type` syntax here.
# https://github.com/pydantic/pydantic/issues/8984
Vars: TypeAlias = list[str]
Expr: TypeAlias = str
Fun: TypeAlias = tuple[Vars, Expr]
IntFun: TypeAlias = Fun
IntPred: TypeAlias = Fun
State: TypeAlias = dict[str, int]


@dataclass
class SynthetizeFunIP:
    conjecture: dp.Policy[dp.Failure | dp.Branch, dp.PromptingPolicy]
    disprove: dp.Policy[dp.Failure | dp.Branch, dp.PromptingPolicy]
    aggregate: dp.PromptingPolicy


@dp.strategy
def synthetize_fun(
    vars: Vars, prop: IntPred
) -> dp.Strategy[Conjecture, SynthetizeFunIP, IntFun]:
    """
    The goal is to synthetize the body of a function f that respects
    properties for all inputs.
    """
    res = yield from make_conjecture(
        cands=conjecture_expr(vars, prop).using(
            lambda p: p.conjecture, SynthetizeFunIP
        ),
        disprove=lambda conj: find_counterexample(vars, prop, conj).using(
            lambda p: p.disprove
        ),
        aggregate=lambda conjs: RemoveDuplicates(conjs).using(
            lambda p: p.aggregate
        ),
    )
    # TODO: prove correctness!
    return (vars, res)


@dp.strategy
def conjecture_expr(
    vars: Vars, prop: IntPred
) -> dp.Strategy[dp.Branch | dp.Failure, dp.PromptingPolicy, Expr]:
    expr = yield from dp.branch(ConjectureExpr(vars, prop).using(one_pp))
    yield from dp.ensure(expr_safe(expr), "Possibly unsafe expression")
    try:
        eval(expr, {v: 0 for v in vars})
    except Exception:
        yield from dp.fail("Invalid expression")
    return expr


@dp.strategy
def find_counterexample(
    vars: Vars, prop: IntPred, expr: Expr
) -> dp.Strategy[dp.Branch | dp.Failure, dp.PromptingPolicy, None]:
    cand = yield from dp.branch(ProposeCex(prop, (vars, expr)).using(one_pp))
    try:
        yield from dp.ensure(
            not check_prop((vars, expr), prop, cand), "Invalid counterexample."
        )
    # Note: it is important to catch Exception instead of having an
    # unqualified `except`. Indeed, doing the former would cause runtime
    # errors such as : "RuntimeError: generator ignored GeneratorExit".
    except Exception:
        yield from dp.fail("Failed to process counterexample.")


@dataclass
class ConjectureExpr(dp.Query[Expr]):
    vars: Vars
    prop: IntPred
    __parser__: ClassVar = dp.raw_yaml


@dataclass
class RemoveDuplicates(dp.Query[Sequence[Expr]]):
    exprs: Sequence[Expr]
    __parser__: ClassVar = dp.raw_yaml


@dataclass
class ProposeCex(dp.Query[State]):
    prop: IntPred
    fun: IntFun
    __parser__: ClassVar = dp.raw_yaml


def expr_safe(expr: Expr) -> bool:
    # Not very reliable! Do not run in production.
    return re.match(r"^[^\._]+$", expr) is not None


def fun_lambda(fun: Fun) -> str:
    args, expr = fun
    args_str = ", ".join(args)
    return f"lambda {args_str}: {expr}"


def check_prop(fun: Fun, prop: IntPred, state: State) -> bool:
    to_eval = f"(lambda F: {prop[1]})({fun_lambda(fun)})"
    return eval(to_eval, state.copy())


def test_expr_safe():
    assert expr_safe("x + y")
    assert not expr_safe("sys.exit()")


def test_check_prop():
    prop: IntPred = (["a", "b"], "F(a, b) == F(b, a) and F(0, 0) == 0")
    fun1: IntFun = (["x", "y"], "x + y")
    fun2: IntFun = (["x", "y"], "x - y")
    assert check_prop(fun1, prop, {"a": 1, "b": 2})
    assert check_prop(fun2, prop, {"a": 0, "b": 0})
    assert not check_prop(fun2, prop, {"a": 1, "b": -1})


#####
##### Iterated Strategies
#####


@dataclass
class PickBoyName(dp.Query[str]):
    # TODO: can we have sequences without causing serialization bugs?
    names: list[str]
    picked_already: list[str]
    __parser__: ClassVar = dp.raw_yaml


@dp.strategy
def pick_boy_name(
    names: Sequence[str], picked_already: Sequence[str]
) -> dp.Strategy[dp.Branch | dp.Failure, dp.PromptingPolicy, str]:
    name = yield from dp.branch(
        PickBoyName(list(names), list(picked_already)).using(one_pp)
    )
    yield from dp.ensure(name in names, "Unavailable name.")
    yield from dp.ensure(name not in picked_already, "Already picked.")
    return name


@dp.strategy
def pick_nice_boy_name(
    names: Sequence[str],
) -> dp.Strategy[dp.Branch | dp.Failure, "PickNiceBoyNameIP", str]:
    name = yield from dp.branch(
        dp.iterated(
            lambda prev: pick_boy_name(names, prev).using(
                lambda ip: ip.pick_boy_name, PickNiceBoyNameIP
            )
        )
    )
    yield from dp.ensure(name == "Jonathan", "You can do better.")
    return name


@dataclass
class PickNiceBoyNameIP:
    pick_boy_name: dp.Policy[dp.Branch | dp.Failure, dp.PromptingPolicy]


#####
##### Testing BFS
#####


# We want to artificially create the following tree:
# The confidences are annotated as costs: cost = -log_10(confidence).
#
#   @ -- 1 (3) -- @ -- 0 (2) -- 11
#   |             |
#   |             + -- 0 (3) -- 12
#   |
#   + -- 0 (3) -- @ -- 0 (1) -- 21
#                 |
#                 + -- 0 (2) -- 22
#
# Confidences in parentheses are priors.
#
# [(1, 1), (2, 1), (2, 2), (1, 2)]


@dataclass
class PickPositiveInteger(dp.Query[int]):
    prev: int | None
    __parser__: ClassVar = dp.raw_yaml


@dp.strategy
def num_confidence(
    prev: int | None, new: int
) -> dp.Strategy[Never, object, float]:
    return 0.1 if (prev, new) == (None, 1) else 1
    yield


@dp.strategy
def generate_pairs() -> dp.Strategy[
    dp.BestFS, dp.PromptingPolicy, tuple[int, int]
]:
    x = yield from dp.bestfs_branch(
        PickPositiveInteger(None)(IP := dp.PromptingPolicy, lambda p: p),
        confidence_priors=lambda _: [1e-3, 1e-3, 0],
    )
    yield from dp.bestfs_factor(
        num_confidence(None, x)(IP, lambda _: (dp.dfs(), ()))
    )
    y = yield from dp.bestfs_branch(
        PickPositiveInteger(x)(IP, lambda p: p),
        confidence_priors=lambda _: (
            [0.01, 0.001, 1e-8] if x == 1 else [0.1, 0.01, 1e-15]
        ),
    )
    yield from dp.bestfs_factor(
        num_confidence(x, y)(IP, lambda _: (dp.dfs(), ())),
    )
    return (x, y)


#####
##### Trivial strategy examples
#####


@dp.strategy
def trivial_strategy() -> dp.Strategy[Never, object, int]:
    return 42
    yield


@dp.strategy
def buggy_strategy() -> dp.Strategy[dp.Failure, object, int]:
    yield from dp.ensure(True, "ok")
    assert False
