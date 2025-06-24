"""
Testing oracular programs in an end-to-end fashion
"""

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, cast

import example_strategies as ex
import pytest

import delphyne as dp

PROMPT_DIR = Path(__file__).parent / "prompts"
CACHE_DIR = Path(__file__).parent / "cache"


def test_query_properties():
    q1 = ex.StructuredOutput(topic="AI")
    assert q1.query_config().force_structured_output
    q2 = ex.MakeSum([1, 2, 3, 4], 7)
    assert not q2.query_config().force_structured_output
    q3 = ex.ProposeArticle(user_name="Alice")
    assert q3.query_config().force_tool_call
    assert len(q3.query_tools()) == 3  # Counting the answer tool


def _eval_query(
    query: dp.Query[object],
    cache_name: str,
    budget: int = 1,
    concurrent: int = 1,
    model_name: dp.StandardModelName = "gpt-4.1-mini",
):
    env = dp.PolicyEnv(demonstration_files=(), prompt_dirs=(PROMPT_DIR,))
    cache = CACHE_DIR / cache_name
    model = dp.CachedModel(dp.standard_model(model_name), cache)
    bl = dp.BudgetLimit({dp.NUM_REQUESTS: budget})
    pp = dp.with_budget(bl) @ dp.few_shot(model, num_concurrent=concurrent)
    stream = query.run_toplevel(env, pp)
    res, _ = dp.collect(stream)
    log = list(env.tracer.export_log())
    print(log)
    return res, log


def _eval_strategy[N: dp.Node, P, T](
    strategy: dp.StrategyInstance[N, P, T],
    policy: Callable[[dp.LLM], dp.Policy[N, P]],
    cache_name: str,
    max_requests: int = 1,
    max_res: int = 1,
    model_name: dp.StandardModelName = "gpt-4.1-mini",
) -> tuple[Sequence[dp.Tracked[T]], str]:
    env = dp.PolicyEnv(prompt_dirs=(), demonstration_files=())
    cache = CACHE_DIR / cache_name
    model = dp.CachedModel(dp.standard_model(model_name), cache)
    stream = strategy.run_toplevel(env, policy(model))
    budget = dp.BudgetLimit({dp.NUM_REQUESTS: max_requests})
    ret, _spent = dp.collect(stream, budget=budget, num_generated=max_res)
    log = list(env.tracer.export_log())
    log_str = "\n".join(e.message for e in log)
    return ret, log_str


def test_concurrent():
    res, _ = _eval_query(
        ex.StructuredOutput(topic="Love"),
        "structured_output_concurrent",
        budget=4,
        concurrent=2,
    )
    assert len(res) == 4


def test_basic_llm_call():
    env = dp.PolicyEnv(demonstration_files=(), prompt_dirs=(PROMPT_DIR,))
    cache = CACHE_DIR / "basic_llm_call"
    model = dp.CachedModel(dp.openai_model("gpt-4.1-mini"), cache)
    pp = dp.few_shot(model)
    bl = dp.BudgetLimit({dp.NUM_REQUESTS: 1})
    policy = dp.take(1) @ (dp.with_budget(bl) @ dp.dfs()), ex.MakeSumIP(pp)
    stream = ex.make_sum([1, 2, 3, 4], 7).run_toplevel(env, policy)
    res, _ = dp.collect(stream)
    # print(list(env.tracer.export_log()))
    assert res


def test_structured_output():
    res, _ = _eval_query(ex.StructuredOutput(topic="AI"), "structured_output")
    assert res
    print(res[0].value)


def test_propose_article_initial_step():
    # Interestingly, the answer content is often empty here despite us
    # explicitly asking for a additional reasoning. Is it because we
    # mandate tool calls? Probably, yes.
    label = "propose_article_initial_step"
    res, _ = _eval_query(ex.ProposeArticle(user_name="Alice"), label)
    v = cast(dp.Response[Any, Any], res[0].value)
    assert isinstance(v, dp.Response)
    assert isinstance(v.parsed, dp.ToolRequests)
    assert isinstance(v.parsed.tool_calls[0], ex.GetUserFavoriteTopic)


def test_assistant_priming():
    res, log = _eval_query(ex.PrimingTest(style="French"), "assistant_priming")
    assert res
    assert len(log[0].metadata["request"]["chat"]) == 3  # type: ignore


def test_interact():
    env = dp.PolicyEnv(demonstration_files=(), prompt_dirs=(PROMPT_DIR,))
    cache = CACHE_DIR / "interact"
    model = dp.CachedModel(dp.openai_model("gpt-4.1-mini"), cache)
    pp = dp.few_shot(model)
    bl = dp.BudgetLimit({dp.NUM_REQUESTS: 2})
    policy = dp.take(1) @ (dp.with_budget(bl) @ dp.dfs()), pp
    stream = ex.propose_article("Jonathan").run_toplevel(env, policy)
    res, _ = dp.collect(stream)
    print(list(env.tracer.export_log()))
    assert res


#####
##### Flags
#####


def test_flags_static():
    f = ex.MethodFlag()
    assert str(f.answer_type()) == "typing.Literal['def', 'alt']"
    assert f.default_answer().content == "def"
    assert len(f.finite_answer_set()) == 2


#####
##### Classification
#####


def _eval_classifier_query(
    query: dp.Query[object],
    cache_name: str,
):
    env = dp.PolicyEnv(demonstration_files=(), prompt_dirs=(PROMPT_DIR,))
    cache = CACHE_DIR / cache_name
    model = dp.CachedModel(dp.openai_model("gpt-4.1-mini"), cache)
    bl = dp.BudgetLimit({dp.NUM_REQUESTS: 1})
    pp = dp.with_budget(bl) @ dp.classify(model)
    stream = query.run_toplevel(env, pp)
    res, _ = dp.collect_with_metadata(stream)
    log = list(env.tracer.export_log())
    print(log)
    assert res
    return res[0][1]


@pytest.mark.parametrize(
    "name, right, wrong",
    [("Jonathan", "common", "rare"), ("X Æ A-Xii:", "rare", "common")],
)
def test_classifiers(name: str, right: str, wrong: str):
    res = _eval_classifier_query(ex.EvalNameRarity(name), "classify")
    assert isinstance(res, dp.LogProbInfo)
    D = {k.content: v for k, v in res.logprobs.items()}
    print(D)
    assert D[right] > D[wrong]


#####
##### Diverse Providers
#####


@pytest.mark.parametrize(
    "model",
    [
        "mistral-small-2503",
        "deepseek-chat",
        # "deepseek-reasoner",  # long latency
    ],
)
def test_provider(model: dp.StandardModelName):
    cache_name = f"provider_{model}"
    query = ex.MakeSum(allowed=[1, 2, 3, 4], goal=5)
    res, _ = _eval_query(query, cache_name, model_name=model)
    assert res
    print(res[0].value)


@pytest.mark.parametrize(
    "model",
    [
        "mistral-small-2503",
        "deepseek-chat",
    ],
)
def test_structured_output_provider(model: dp.StandardModelName):
    cache_name = f"provider_structured_{model}"
    query = ex.StructuredOutput(topic="AI")
    _, _ = _eval_query(query, cache_name, model_name=model)
    # We explicitly do not ensure that there is a result since DeepSeek
    # will likely fail to respect the schema without further prompting.


#####
##### Abduction
#####


# @pytest.mark.skip("todo")
def test_abduction():
    """
    Testing on the following market:

        S1: -> A
        S2: -> B
        S3: A,D -> C
        S4: B -> E
        S5: A,E -> C
        S6: C -> F

    The goal is to obtain F.
    """
    market: ex.Market = [
        ex.MarketMember("R1", [], "A"),
        ex.MarketMember("R2", [], "B"),
        ex.MarketMember("R3", ["A", "D"], "C"),
        ex.MarketMember("R4", ["B"], "E"),
        ex.MarketMember("R5", ["A", "E"], "C"),
        ex.MarketMember("R6", ["C"], "F"),
    ]
    strategy = ex.obtain_item(market, "F")
    policy = partial(ex.obtain_item_policy, num_concurrent=1)
    cache_name = "abduction"
    res, log = _eval_strategy(
        strategy, policy, cache_name, max_requests=10, max_res=1
    )
    # assert res
    print(res)
    print()
    print(log)
