"""
Testing oracular programs in an end-to-end fashion
"""

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
