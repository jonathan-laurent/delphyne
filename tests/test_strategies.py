"""
Simple test strategies.

The strategies defined in this file are used to test `StrategyTree` but
also to test the server (see `test_server`).
"""

import re
import textwrap
from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, TypeAlias, TypeGuard, cast

from delphyne.core.inspect import (
    underlying_strategy_param_type,
    underlying_strategy_return_type,
)
from delphyne.core.queries import ParseError
from delphyne.core.refs import ChoiceOutcomeRef, NodeId
from delphyne.core.strategies import StrategyTree
from delphyne.core.tracing import Outcome, Value
from delphyne.core.trees import Navigation, Node, Strategy, Tree
from delphyne.stdlib.composable_policies import (
    PolicyElement,
    compose,
    composed_policy,
)
from delphyne.stdlib.dsl import (
    GeneratorConvertible,
    convert_to_generator,
    convert_to_parametric_generator,
    strategy,
)
from delphyne.stdlib.generators import GenEnv, Generator, GenRet
from delphyne.stdlib.nodeclasses import nodeclass
from delphyne.stdlib.nodes import (
    Branch,
    Branching,
    Failure,
    Run,
    branch,
    ensure,
    fail,
    run,
)
from delphyne.stdlib.search.bfs import BFS, bfs, bfs_branch, bfs_factor
from delphyne.stdlib.search.composable_basic import handle_failure, handle_run
from delphyne.stdlib.search.dfs import dfs
from delphyne.stdlib.search.iterated import iterated
from delphyne.stdlib.search_envs import Params
from delphyne.stdlib.structured import StructuredQuery
from delphyne.utils.typing import NoTypeInfo


#####
##### Strategy Example: make_sum
#####


@dataclass
class SearchParams(Params):
    branching_factor_small: int = 2
    branching_factor_large: int = 5


def BRANCH_SMALL(params: SearchParams) -> int:
    return params.branching_factor_small


@strategy(dfs)
def make_sum(
    allowed: list[int], goal: int
) -> Branching[SearchParams, list[int]]:
    xs = yield from branch(MakeSum(allowed, goal), BRANCH_SMALL)
    yield from ensure(all(x in allowed for x in xs), "forbidden-num")
    yield from ensure(sum(xs) == goal, "wrong-sum")
    return xs


@dataclass
class MakeSum(StructuredQuery[Params, list[int]]):
    allowed: list[int]
    goal: int

    def system_message(self, params: object) -> str:
        return dedent(
            """
            Please output a list of numbers that sums up to the provided goal.
            All numbers must belong to the `allowed` list.
            Repetitions are allowed.
            """
        )


def dedent(s: str) -> str:
    return textwrap.dedent(s).strip() + "\n"


#####
##### Adding support for conjecture nodes
#####


@nodeclass(frozen=True)
class Conjecture[P, T](Node):
    candidate: Generator[P, T]
    disprove: Callable[[Outcome[T]], Generator[P, None]]
    aggregate: Callable[[tuple[Outcome[T], ...]], Generator[P, Sequence[T]]]

    def navigate(self) -> Navigation:
        return (yield self.candidate)


def make_conjecture[P: Params, T](
    candidate: GeneratorConvertible[P, T],
    disprove: Callable[[T], GeneratorConvertible[P, None]],
    aggregate: Callable[[tuple[T, ...]], GeneratorConvertible[P, Sequence[T]]],
) -> Strategy[Conjecture[P, Any], T]:  # fmt: skip
    cand = yield Conjecture(
        convert_to_generator(candidate),
        convert_to_parametric_generator(disprove),
        convert_to_parametric_generator(aggregate),
    )
    return cast(T, cand)


def handle_conjecture[P](params: P) -> PolicyElement[Conjecture[P, Any]]:

    async def visit[N: Node, T](
        env: GenEnv,
        node: Conjecture[P, Any],
        tree: Tree[N, T],
        recurse: Callable[[GenEnv, Tree[N, T]], GenRet[T]]
    ) -> GenRet[T]:  # fmt: skip
        # TODO: we are ignoring `aggregate` and `disprove` here.
        # We are behaving like `run` right now.
        async for resp in node.candidate(env, tree, params):
            if not resp.items:
                yield resp
            if resp.items:
                async for ret in recurse(env, tree.child(resp.items[0])):
                    yield ret
                return

    def guard(obj: object) -> TypeGuard[Conjecture[P, Any]]:
        return isinstance(obj, Conjecture)

    return guard, visit


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


type Synth[P, T] = Strategy[Conjecture[P, Any] | Run[P] | Failure, T]
handle_synth = compose(handle_conjecture, compose(handle_run, handle_failure))


@strategy(composed_policy(handle_synth))
def synthetize_fun(vars: Vars, prop: IntPred) -> Synth[Params, IntFun]:
    """
    The goal is to synthetize the body of a function f that respects
    properties for all inputs.
    """
    res = yield from make_conjecture(
        conjecture_expr(vars, prop),
        disprove=partial(find_counterexample, vars, prop),
        aggregate=RemoveDuplicates,
    )
    # TODO: prove correctness!
    return (vars, res)


@strategy(dfs)
def conjecture_expr(vars: Vars, prop: IntPred) -> Branching[Params, Expr]:
    expr = yield from branch(ConjectureExpr(vars, prop))
    yield from ensure(expr_safe(expr), "Possibly unsafe expression")
    try:
        eval(expr, {v: 0 for v in vars})
    except Exception:
        yield from fail("Invalid expression")
    return expr


@strategy(composed_policy(compose(handle_run, handle_failure)))
def find_counterexample(
    vars: Vars, prop: IntPred, expr: Expr
) -> Strategy[Run[Params] | Failure, None]:
    cand = yield from run(ProposeCex(prop, (vars, expr)))
    try:
        yield from ensure(
            not check_prop((vars, expr), prop, cand), "Invalid counterexample."
        )
    # Note: it is important to catch Exception instead of having an
    # unqualified `except`. Indeed, doing the former would cause runtime
    # errors such as : "RuntimeError: generator ignored GeneratorExit".
    except Exception:
        yield from fail("Failed to process counterexample.")


@dataclass
class ConjectureExpr(StructuredQuery[Params, Expr]):
    vars: Vars
    prop: IntPred


@dataclass
class RemoveDuplicates(StructuredQuery[Params, tuple[Expr, ...]]):
    exprs: tuple[Expr, ...]


@dataclass
class ProposeCex(StructuredQuery[Params, State]):
    prop: IntPred
    fun: IntFun


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
##### Iterated strategies
#####


@dataclass
class PickBoyName(StructuredQuery[Params, str]):
    names: list[str]
    picked_already: list[str]


@strategy(dfs)
def pick_boy_name(
    names: Sequence[str], picked_already: Sequence[str]
) -> Branching[Params, str]:
    name = yield from branch(PickBoyName(list(names), list(picked_already)))
    yield from ensure(name in names, "Unavailable name.")
    yield from ensure(name not in picked_already, "Already picked.")
    return name


@strategy(dfs)
def pick_nice_boy_name(names: Sequence[str]) -> Branching[Params, str]:
    name = yield from branch(iterated(partial(pick_boy_name, names)))
    yield from ensure(name == "Jonathan", "You can do better.")
    return name


#####
##### Testing BFS
#####


# We want to artificially create the following tree:
# The confidences are annotated as costs: cost = -log_10(confidence).
#
#   @ -- (3) 1 -- @ -- (2) -- 11
#   |             |
#   |             + -- (3) -- 12
#   |
#   + -- (3) 0 -- @ -- (1) -- 21
#                 |
#                 + -- (2) -- 22
#
# Confidences in parentheses are priors.

# Let's start with no penalty for creating children. The order in which
# the elements should be found is: 11, 21, 22, 12. However, if we put a
# strong penalty for discovering 12, it will not be discovered first
# anymore.


@dataclass
class PickPositiveInteger(StructuredQuery[Params, int]):
    prev: int | None

    def system_message(self, params: Params) -> str:
        return "Please pick a positive integer."


@strategy(dfs)
def num_confidence(prev: int | None, new: int) -> Branching[Params, float]:
    return 0.1 if (prev, new) == (None, 1) else 1
    yield


@strategy(bfs)
def generate_pairs() -> BFS[Params, tuple[int, int]]:
    """ """
    x = yield from bfs_branch(
        PickPositiveInteger(None),
        confidence_priors=lambda _: [1e-3, 1e-3, 0],
        param_type=Params,
    )
    yield from bfs_factor(num_confidence(None, x))
    y = yield from bfs_branch(
        PickPositiveInteger(x),
        confidence_priors=lambda _: (
            [0.01, 0.001, 1e-8] if x == 1 else [0.1, 0.01, 1e-15]
        ),
        param_type=Params,
    )
    yield from bfs_factor(num_confidence(x, y))
    return (x, y)


#####
##### Trivial strategy examples
#####


@strategy()
def trivial_strategy() -> Branching[SearchParams, int]:
    return 42
    yield


@strategy()
def buggy_strategy() -> Branching[SearchParams, int]:
    yield from ensure(True, "ok")
    assert False


#####
##### Tests
#####


def dummy_outcome(x: object) -> Value:
    choice_ref = ("__dummy__", ())
    return Outcome(x, ChoiceOutcomeRef(choice_ref, NodeId(0)), NoTypeInfo())


def test_nodeclass():
    assert len(getattr(Conjecture, "__subchoices__")) == 3
    assert getattr(Conjecture, "__primary_choice__") == "candidate"
    assert getattr(Conjecture, "__match_args__") == ()


def test_make_sum():
    root = StrategyTree.new(make_sum([4, 6, 2, 9], 11))
    assert isinstance(root.node, Branch)
    tree = root.child(dummy_outcome([4, 6]))
    assert isinstance(tree.node, Failure) and tree.node.message == "wrong-sum"
    tree = root.child(dummy_outcome([11]))
    assert (
        isinstance(tree.node, Failure) and tree.node.message == "forbidden-num"
    )
    tree = root.child(dummy_outcome([2, 9]))


def test_type_inspection():
    assert underlying_strategy_return_type(make_sum) == list[int]
    assert underlying_strategy_return_type(make_sum([1], 1)) == list[int]
    assert underlying_strategy_param_type(make_sum) == SearchParams
    assert underlying_strategy_param_type(synthetize_fun) == Params


def test_structured_queries():
    query = MakeSum(allowed=[1], goal=1)
    assert query.name() == "MakeSum"
    assert query.return_type() == list[int]
    assert query.serialize_args() == {"allowed": [1], "goal": 1}
    assert query.parse_answer("[2, 3, 4]") == [2, 3, 4]
    # Pydantic validation error
    assert isinstance(query.parse_answer("[2, 3 4]"), ParseError)
    # Yaml parser error
    assert isinstance(query.parse_answer("[2, 3 4"), ParseError)
