"""
Simple example strategies.

The strategies defined in this file are used to test `Tree` but also to
test the server (see `test_server`).
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Literal, Never, Sequence, cast, override

# Reexporting untyped strategies
# ruff: noqa: F401
from example_strategies_untyped import (
    trivial_untyped_strategy,  # type: ignore
)

import delphyne as dp
from delphyne import Branch, Fail, IPDict, Strategy, strategy

#####
##### MakeSum
#####


@dataclass
class MakeSum(dp.Query["list[int]"]):
    allowed: list[int]
    goal: int
    __parser__ = dp.get_text.yaml


@dataclass
class MakeSumIP:
    make_sum: dp.PromptingPolicy


@strategy
def make_sum(
    allowed: list[int], goal: int
) -> "Strategy[Branch | Fail, MakeSumIP, list[int]]":
    # We are testing that string type annotations work too.
    xs = yield from dp.branch(
        cands=MakeSum(allowed, goal).using(lambda p: p.make_sum),
        inner_policy_type=MakeSumIP,
    )
    yield from dp.ensure(all(x in allowed for x in xs), label="forbidden_num")
    yield from dp.ensure(sum(xs) == goal, label="wrong_sum")
    return xs


@dp.ensure_compatible(make_sum)
def make_sum_policy(model_name: str = "gpt-4o-mini"):
    model = dp.openai_model(model_name)
    return dp.dfs() & MakeSumIP(dp.few_shot(model))


#####
##### MakeSum with IPDict
#####


@strategy
def make_sum_dict_ip(
    allowed: list[int], goal: int
) -> Strategy[Branch | Fail, IPDict, list[int]]:
    xs = yield from dp.branch(MakeSum(allowed, goal).using(...))
    yield from dp.ensure(all(x in allowed for x in xs), label="forbidden_num")
    yield from dp.ensure(sum(xs) == goal, label="wrong_sum")
    return xs


def make_sum_dict_ip_policy(model: dp.LLM):
    return dp.dfs() & {"MakeSum": dp.few_shot(model)}


#####
##### Adding support for conjecture nodes
#####


@dataclass(frozen=True)
class Conjecture(dp.Node):
    """
    An example effect `Conjecture`. See `make_conjecture`.
    """

    cands: dp.OpaqueSpace[Any, Any]
    disprove: Callable[[dp.Tracked[Any]], dp.OpaqueSpace[Any, None]]
    aggregate: Callable[
        [tuple[dp.Tracked[Any], ...]], dp.OpaqueSpace[Any, Sequence[Any]]
    ]

    @override
    def navigate(self) -> dp.Navigation:
        return (yield self.cands)

    @override
    def primary_space(self):
        return self.cands


def make_conjecture[P, T](
    cands: dp.Opaque[P, T],
    disprove: Callable[[T], dp.Opaque[P, None]],
    aggregate: Callable[[tuple[T, ...]], dp.Opaque[P, Sequence[T]]],
    inner_policy_type: type[P] | None = None,
) -> Strategy[Conjecture, P, T]:
    """
    Triggering function for the `Conjecture` effect.

    This acts like the `branch` function, except that an additional
    function for disproving candidates is provided, as well as an
    aggregation function that combines several candidates into a list of
    candidates (e.g. removing semantic duplicates).
    """
    cand = yield dp.spawn_node(
        Conjecture, cands=cands, disprove=disprove, aggregate=aggregate
    )
    return cast(T, cand)


@dp.search_policy
def just_guess[P, T](
    tree: dp.Tree[Conjecture, P, T], env: dp.PolicyEnv, policy: P
) -> dp.StreamGen[T]:
    """
    Do a DFS, treating conjecture nodes as simple branching nodes.
    """
    match tree.node:
        case dp.Success(x):
            yield dp.Solution(x)
        case Conjecture(candidate):
            rec = candidate.stream(env, policy).bind(
                lambda y: just_guess()(
                    tree.child(y.tracked), env, policy
                ).gen()
            )
            yield from rec.gen()


#####
##### Utility: single prompting policy
#####


def one_pp(p: dp.PromptingPolicy) -> dp.PromptingPolicy:
    return p


#####
##### A more complex strategy with counterexamples
#####


type Vars = list[str]
type Expr = str
type Fun = tuple[Vars, Expr]
type IntFun = Fun
type IntPred = Fun
type State = dict[str, int]


@dataclass
class SynthetizeFunIP:
    conjecture: dp.Policy[Fail | Branch, dp.PromptingPolicy]
    disprove: dp.Policy[Fail | Branch, dp.PromptingPolicy]
    aggregate: dp.PromptingPolicy


@strategy
def synthetize_fun(
    vars: Vars, prop: IntPred
) -> Strategy[Conjecture, SynthetizeFunIP, IntFun]:
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


@strategy
def conjecture_expr(
    vars: Vars, prop: IntPred
) -> Strategy[Branch | Fail, dp.PromptingPolicy, Expr]:
    expr = yield from dp.branch(ConjectureExpr(vars, prop).using(one_pp))
    yield from dp.ensure(expr_safe(expr), label="possibly-unsafe-expr")
    try:
        eval(expr, {v: 0 for v in vars})
    except Exception:
        yield from dp.fail(label="invalid-expr")
    return expr


@strategy
def find_counterexample(
    vars: Vars, prop: IntPred, expr: Expr
) -> Strategy[Branch | Fail, dp.PromptingPolicy, None]:
    cand = yield from dp.branch(ProposeCex(prop, (vars, expr)).using(one_pp))
    try:
        yield from dp.ensure(
            not check_prop((vars, expr), prop, cand),
            label="invalid_counterexample",
        )
    # Note: it is important to catch Exception instead of having an
    # unqualified `except`. Indeed, doing the former would cause runtime
    # errors such as : "RuntimeError: generator ignored GeneratorExit".
    except Exception:
        yield from dp.fail(label="failed_to_process_counterexample")


@dataclass
class ConjectureExpr(dp.Query[Expr]):
    vars: Vars
    prop: IntPred
    __parser__ = dp.get_text.yaml


@dataclass
class RemoveDuplicates(dp.Query[Sequence[Expr]]):
    exprs: Sequence[Expr]
    __parser__ = dp.get_text.yaml


@dataclass
class ProposeCex(dp.Query[State]):
    prop: IntPred
    fun: IntFun
    __parser__ = dp.get_text.yaml


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
##### Iteration
#####


@dataclass
class PickBoyName(dp.Query[str]):
    names: Sequence[str]
    picked_already: Sequence[str]
    __parser__ = dp.get_text.yaml


@strategy
def pick_boy_name(
    names: Sequence[str], picked_already: Sequence[str] | None
) -> Strategy[Branch | Fail, dp.PromptingPolicy, tuple[str, Sequence[str]]]:
    if picked_already is None:
        picked_already = []
    assert not isinstance(picked_already, str)
    name = yield from dp.branch(
        PickBoyName(names, picked_already).using(one_pp)
    )
    yield from dp.ensure(name in names, label="unavailable-name.")
    yield from dp.ensure(name not in picked_already, label="already-picked")
    return name, [*picked_already, name]


@strategy
def pick_nice_boy_name(
    names: Sequence[str],
) -> Strategy[Branch | Fail, "PickNiceBoyNameIP", str]:
    name = yield from dp.branch(
        dp.iterate(
            lambda prev: pick_boy_name(names, prev).using(
                lambda ip: ip.pick_boy_name, PickNiceBoyNameIP
            )
        )
    )
    yield from dp.ensure(name == "Jonathan", message="You can do better.")
    return name


@dataclass
class PickNiceBoyNameIP:
    pick_boy_name: dp.Policy[Branch | Fail, dp.PromptingPolicy]


#####
##### Testing BFS
#####


# We want to artificially create the following tree:
# The confidences are annotated as costs: cost = -log_10(confidence).
#
#    @ -- 1 (3) -- @ -- 1 (1) -- 11
#    |             |
#    |             + -- 1 (2) -- 12
#    |
#    + -- 0 (3) -- @ -- 0 (1) -- 21
#                  |
#                  + -- 0 (2) -- 22
#
# Confidences in parentheses are priors.
# Note that the values at the second level do not matter (only the prior).
#
# [(1, 1), (2, 1), (2, 2), (1, 2)]


@dataclass
class PickPositiveInteger(dp.Query[int]):
    prev: int | None
    __parser__ = dp.get_text.yaml


@strategy
def num_confidence(
    prev: int | None, new: int
) -> Strategy[Never, object, float]:
    # Depth 0
    if prev is None:
        return 0.1 if new == 1 else 1
    # Depth 1
    return 0.1 if prev == 1 else 1
    yield


@strategy
def generate_pairs() -> Strategy[
    Branch | dp.Factor | Fail, dp.PromptingPolicy, tuple[int, int]
]:
    x = yield from dp.branch(
        PickPositiveInteger(None)
        .using(lambda p: p, dp.PromptingPolicy)
        .tagged("first")
    )
    yield from dp.factor(
        num_confidence(None, x)
        .using(lambda _: dp.dfs() & None, dp.PromptingPolicy)
        .tagged("first"),
        lambda _: lambda f: f,
    )
    y = yield from dp.branch(
        PickPositiveInteger(x)
        .using(lambda p: p, dp.PromptingPolicy)
        .tagged("second"),
    )
    yield from dp.factor(
        num_confidence(x, y)
        .using(lambda _: (dp.dfs() & None), dp.PromptingPolicy)
        .tagged("second"),
        lambda _: lambda f: f,
    )
    return (x, y)


@dp.ensure_compatible(generate_pairs)
def generate_pairs_policy(pp: dp.PromptingPolicy):
    def child_prior(depth: int, num_prev: int):
        if depth == 0:
            return 0 if num_prev >= 2 else 1e-3
        if depth == 1:
            return 1e-15 if num_prev >= 2 else 1e-2 if num_prev >= 1 else 1e-1
        assert False

    bestfs = dp.best_first_search(child_confidence_prior=child_prior)
    return bestfs & pp


#####
##### Test cached computations
#####


def expensive_computation(
    n: int, timeout: float | None = None
) -> tuple[int, int]:
    if timeout is not None:
        print(f"Timeout set: {timeout}")
    return (n, n + 1)


@strategy
def test_cached_computations(
    n: int,
) -> Strategy[dp.Compute, object, int]:
    a, b = yield from dp.compute(expensive_computation)(n)
    c, d = yield from dp.compute(expensive_computation)(n)
    e, f = yield from dp.compute(expensive_computation)(n + 1)
    return a * b + c * d + e * f


@dp.ensure_compatible(test_cached_computations)
def test_cached_computations_policy():
    return dp.dfs() @ dp.elim_compute() & None


#####
##### Trivial strategy examples
#####


@strategy
def trivial_strategy() -> Strategy[Never, object, int]:
    return 42
    yield


@strategy
def buggy_strategy() -> Strategy[Fail, object, int]:
    yield from dp.ensure(True, label="unreachable")
    assert False


#####
##### Structured output and tool use
#####


@dataclass
class Article:
    title: str
    authors: list[str]


# string type annotations are supported.
@dataclass
class StructuredOutput(dp.Query["Article"]):
    """
    Generate an article on the given topic. Answer as a JSON object.
    """

    topic: str


@dataclass
class GetUserFavoriteTopic(dp.AbstractTool[str]):
    """
    Retrieve the favorite topic of a given user.
    """

    user_name: str


@dataclass
class Calculator(dp.AbstractTool[str]):
    """
    Compute the value of a numerical Python expression.
    """

    expr: str


@dataclass
class ProposeArticle(
    dp.Query[dp.Response[Article, GetUserFavoriteTopic | Calculator]]
):
    user_name: str
    prefix: dp.AnswerPrefix = ()

    __parser__ = dp.final_tool_call.response

    __system_prompt__: ClassVar[str] = """
        Find the user's tastes and propose an article for them.
        Provide your final answer by calling the `Article` tool.
        Please carefully explain your reasoning before calling any tool.
        """


@strategy
def propose_article(
    user_name: str,
) -> Strategy[Branch, dp.PromptingPolicy, Article]:
    article = yield from dp.interact(
        step=lambda pre, _: ProposeArticle(user_name, pre).using(
            dp.ambient_pp
        ),
        process=lambda x, _: dp.const_space(x),
        tools={GetUserFavoriteTopic: (lambda _: dp.const_space("Soccer"))},
    )
    return article


@dataclass
class ProposeArticleStructured(
    dp.Query[dp.Response[Article, GetUserFavoriteTopic | Calculator]]
):
    user_name: str
    prefix: dp.AnswerPrefix = ()

    __parser__ = dp.structured.response

    __system_prompt__: ClassVar[str] = """
        Find the user's tastes and propose an article for them.
        """


@strategy
def propose_article_structured(
    user_name: str,
) -> Strategy[Branch, dp.PromptingPolicy, Article]:
    article = yield from dp.interact(
        step=lambda pre, _: ProposeArticleStructured(user_name, pre).using(
            dp.ambient_pp
        ),
        process=lambda x, _: dp.const_space(x),
        tools={GetUserFavoriteTopic: (lambda _: dp.const_space("Soccer"))},
    )
    return article


def propose_article_policy(
    model: dp.LLM,
) -> dp.Policy[Branch, dp.PromptingPolicy]:
    # Valid for both `propose_article` and `propose_article_structured`
    return dp.dfs(max_branching=1) & dp.few_shot(model)


#####
##### Assistant Priming
#####


@dataclass
class PrimingTest(dp.Query[list[str]]):
    """
    Generate a list of nice baby names in the given style.

    End your answer with a triple-backquoted code block containing a
    list of strings as a YAML object.
    """

    style: str

    __parser__ = dp.last_code_block.yaml

    __instance_prompt__: ClassVar[str] = """
    Style: {{query.style}}
    !<assistant>
    Here are _exactly_ 4 baby boy names in this style (and nothing else):
    """


#####
##### Flags
#####


@dataclass
class MethodFlag(dp.FlagQuery[Literal["def", "alt"]]):
    pass


@strategy
def pick_flag() -> Strategy[dp.Flag[MethodFlag], object, int]:
    match (yield from dp.get_flag(MethodFlag)):
        case "def":
            return 0
        case "alt":
            return 1


#####
##### Classification
#####


@dataclass
class EvalNameRarity(dp.Query[Literal["common", "rare"]]):
    """
    Classify the user name as either "common" or "rare".
    Just answer with one of those words and nothing else.
    """

    user_name: str
    __parser__ = dp.first_word


#####
##### Abduction
#####


@dataclass
class MarketMember:
    name: str
    asked_items: list[str]
    offered_item: str


type Market = list[MarketMember]


@dataclass
class Exchange:
    person: str


@dataclass
class ItemsToFind:
    items: Sequence[str]


@dataclass
class ObtainItem(dp.Query[ItemsToFind]):
    """
    You are in a market and you are presented with a list of sellers.
    Each seller offers to exchange a specific item for a list of other
    items (or for free if the list is empty). You also possess a number
    of items and are interested in obtaining a new one.

    Find all vendors selling the item that you want and answer with a
    list of all the things these vendors want that you do not have
    already.
    """

    market: Market
    possessed_items: Sequence[str]
    item: str


@dataclass
class AcquisitionFeedback:
    item: str
    possessed: Sequence[str]


def _trade_with(
    member: MarketMember,
    possessed: Sequence[tuple[str, list[Exchange]]],
    wanted_item: str,
) -> list[Exchange] | None:
    if member.offered_item != wanted_item:
        return None
    exchanges: list[Exchange] = []
    for asked in member.asked_items:
        for it, xs in possessed:
            if it == asked:
                exchanges += xs
                break
        else:
            return None
    exchanges.append(Exchange(person=member.name))
    return exchanges


def try_obtain_item(
    market: Market, item: str, possessed: Sequence[tuple[str, list[Exchange]]]
) -> dp.AbductionStatus[AcquisitionFeedback, list[Exchange]]:
    # If the item is already possessed, we are done.
    for it, xs in possessed:
        if it == item:
            return ("proved", xs)
    # Otherwise, if we can obtain the item in a single exchange we are
    # still good.
    for member in market:
        attempt = _trade_with(member, possessed, item)
        if attempt is not None:
            return ("proved", attempt)
    # If no one offers the item, there is no chance to obtain it.
    if not any(item == member.offered_item for member in market):
        return ("disproved", None)
    return (
        "feedback",
        AcquisitionFeedback(item, [p[0] for p in possessed]),
    )


@strategy
def obtain_item(
    market: Market, goal: str
) -> Strategy[dp.Abduction | dp.Message, dp.PromptingPolicy, list[Exchange]]:
    IP = dp.PromptingPolicy
    exchanges = yield from dp.abduction(
        prove=lambda possessed, item: (
            dp.const_space(try_obtain_item(market, item or goal, possessed))
        ),
        suggest=lambda f: (
            dp.map_space(
                ObtainItem(market, f.possessed, f.item).using(lambda p: p),
                lambda x: x.items,
            )
        ),
        search_equivalent=lambda its, it: dp.const_space(None),
        redundant=lambda its, it: dp.const_space(False),
        inner_policy_type=IP,
    )
    yield from dp.message("Success!")
    return exchanges


def obtain_item_policy(model: dp.LLM, num_completions: int = 1):
    pp = dp.take(num_completions) @ dp.few_shot(
        model, num_completions=num_completions
    )
    return dp.abduct_and_saturate(log_steps="info") @ dp.elim_messages() & pp


#####
##### Embedded Trees and Transformers
#####


@strategy
def recursive_joins(
    depth: int,
) -> Strategy[
    dp.Join | dp.Message | dp.Compute | dp.Flag[MethodFlag],
    dp.NodeMeta,
    int,
]:
    if depth == 0:
        flag = yield from dp.get_flag(MethodFlag)
        v, _ = yield from dp.compute(expensive_computation)(1)
        return v if flag == "def" else 0
    else:
        yield from dp.message(f"Recursing at depth {depth}")
        res = yield from dp.join(
            [recursive_joins(depth - 1)] * 2, meta=lambda p: p
        )
        return sum(res)


def recursive_joins_policy():
    from delphyne.stdlib.search import recursive_search as dprs

    sp = (
        dprs.recursive_search()
        @ dp.elim_messages()
        @ dp.elim_compute(override_args={"timeout": 1.0})
        @ dp.elim_flag(MethodFlag, "def")
    )
    return sp & dprs.OneOfEachSequentially()


#####
##### DictIPs
#####


@dataclass
class GenerateNumber(dp.Query[int]):
    """
    Generate a number between min_val and max_val (inclusive). Just
    answer with the number and nothing else.
    """

    min_val: int
    max_val: int
    __parser__ = dp.get_text.yaml


@dataclass
class GenerateNumberIP:
    generate_number: dp.PromptingPolicy


@strategy
def generate_number(
    min_val: int, max_val: int
) -> Strategy[Branch | Fail, IPDict, int]:
    num = yield from dp.branch(GenerateNumber(min_val, max_val).using(...))
    yield from dp.ensure(min_val <= num <= max_val, "number_out_of_range")
    return num


@strategy
def dual_number_generation() -> Strategy[
    Branch | Fail, IPDict, tuple[int, int]
]:
    low_num = yield from dp.branch(
        generate_number(1, 50).using(...).tagged("low")
    )
    high_num = yield from dp.branch(
        generate_number(51, 100).using(...).tagged("high")
    )
    yield from dp.ensure(low_num < high_num, label="numbers_not_ordered")
    return (low_num, high_num)


@dp.ensure_compatible(dual_number_generation)
def dual_number_generation_policy(model: dp.LLM, shared: bool):
    sub = dp.dfs() & {"GenerateNumber": dp.few_shot(model)}
    if shared:
        ip = {"generate_number": sub}
    else:
        ip = {
            "generate_number&low": sub,
            "generate_number&high": sub,
        }
    return dp.dfs() & ip


@dp.ensure_compatible(dual_number_generation)
def dual_number_generation_parallel_policy(model: dp.LLM):
    sub = dp.take(2) @ dp.dfs() & {"GenerateNumber": dp.few_shot(model)}
    ip = {
        "generate_number&low": sub,
        "generate_number&high": sub,
    }
    return dp.par_dfs() & ip


#####
##### Testing Imperative Strategies
#####


class DummyChoice(dp.Query[bool]):
    __parser__ = dp.get_text.yaml


@strategy
def imperative_strategy() -> Strategy[Branch | Fail, IPDict, None]:
    goal = 3
    allowed = [1, 2, 3]
    res = yield from dp.branch(MakeSum(allowed, goal).using(...))
    goal += 4
    # `allowed` can be safely mutated, without the query getting
    # affected. Indeed, the strategy's coroutine is rerun from the start
    # each time and so the `allowed` list before and after `yield from`
    # are not the same.
    allowed.append(4)
    #  Values returned by `yield` (actions) can be safely mutated since
    #  they are deep-copied in the tree internal structure. Without such
    #  copy, the second explored branch of the choice below would fail
    #  the test since `4` would have been appended twice to `res`.
    res.append(4)
    _choice = yield from dp.branch(DummyChoice().using(...))
    yield from dp.ensure(sum(res) == goal, label="wrong_sum")
    return


#####
##### Mode Handling
#####


@dataclass
class Dish:
    name: str
    ingredients: list[str]


@dataclass
class GetFavoriteDish(dp.Query[Dish]):
    """
    Describe the favorite dish of the provided user.

    {% if mode == "cot" %}
    Finish your answer with a triple-quoted YAML block containing a
    single object with fields `name` (str) and `ingredients` (list[str]).
    {% else %}
    Just answer with a JSON object.
    {% endif %}
    """

    user: str

    __modes__ = ["cot", "direct"]
    __parser__ = {"cot": dp.last_code_block.yaml, "direct": dp.structured}


#####
##### Error Wrapping and Hindsight Feedback
#####


def parse_83(ans: str) -> int:
    if ans == "83":
        return 83
    raise dp.ParseError(
        description="Expected '83'. Please answer with 83 instead."
    )


@dataclass
class AskNumber(dp.Query[dp.Response[int | dp.WrappedParseError, Never]]):
    """
    Answer with a number and nothing else.
    """

    prefix: dp.AnswerPrefix = ()

    __parser__ = dp.get_text.map(parse_83).wrap_errors.response

    @override
    def hindsight_answer(self, feedback: object):
        assert isinstance(feedback, int)
        return dp.Answer(None, str(feedback))


@strategy
def get_magic_number() -> Strategy[
    Branch | dp.Hindsight, dp.PromptingPolicy, int
]:
    ret = yield from dp.interact(
        step=lambda pre, _: AskNumber(pre).using(dp.ambient_pp),
        process=lambda x, _: dp.const_space(x),
    )
    yield from dp.hindsight(
        AskNumber(()), feedback=ret, as_parsed_answer=False
    )
    return ret


@dp.ensure_compatible(get_magic_number)
def get_magic_number_policy(model: dp.LLM, no_wrap: bool):
    sp = dp.dfs() @ dp.elim_hindsight()
    pp = dp.few_shot(model, no_wrap_parse_errors=no_wrap)
    return sp & pp


@dp.ensure_compatible(get_magic_number)
def get_magic_number_default_policy(model: str = "gpt-5-nano"):
    return get_magic_number_policy(dp.standard_model(model), no_wrap=False)


#####
##### Universal Queries
#####


@dp.strategy
def make_sum_using_guess(
    allowed: list[int], goal: int
) -> Strategy[Branch | Fail, IPDict, list[int]]:
    """
    Given a list of numbers and a target number, return a sub-list whose
    elements sum up to the target.
    """
    sub = yield from dp.guess(list[int], using=[allowed, goal])
    yield from dp.ensure(all(x in allowed for x in sub), label="forbidden_num")
    yield from dp.ensure(sum(sub) == goal, label="wrong_sum")
    return sub


@dp.ensure_compatible(make_sum_using_guess)
def make_sum_using_guess_policy(model: dp.LLM):
    return dp.dfs() & {"sub": dp.few_shot(model)}


#####
##### Binarize Values, Iterative Mode for `few_shot`
#####


@dataclass
class NumberOutput:
    number: int


@dataclass
class GuessZeroAndThenOne(dp.Query[NumberOutput]):
    __system_prompt__: ClassVar[str] = "Answer with a number."
    __instance_prompt__: ClassVar[str] = "Start trying to answer `0`."
    __more_prompt__: ClassVar[str] = "Now try to answer `1`."


@dp.strategy
def guess_zero_and_then_one() -> Strategy[
    dp.Value | Branch, dp.PromptingPolicy, int
]:
    res = yield from dp.branch(GuessZeroAndThenOne().using(dp.ambient_pp))
    num = res.number
    yield from dp.value(dp.const_space(float(num == 1)), lambda _: lambda x: x)
    return num


@dp.ensure_compatible(guess_zero_and_then_one)
def guess_zero_and_then_one_policy(model: dp.LLM):
    sp = dp.dfs() @ dp.binarize_values(threshold=0.5)
    return sp & dp.few_shot(model, iterative_mode=True)


#####
##### Data
#####


@dp.strategy
def strategy_loading_data(key: str) -> dp.Strategy[dp.Data, None, str]:
    articles = yield from dp.load_data([("articles", key)], type=Article)
    return articles[0].title


@dp.ensure_compatible(strategy_loading_data)
def strategy_loading_data_policy():
    return dp.dfs() @ dp.elim_data() & None
