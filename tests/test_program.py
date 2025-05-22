"""
Testing oracular programs in an end-to-end fashion
"""

from pathlib import Path
from typing import Any, cast

import example_strategies as ex

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


def _eval_query(query: dp.Query[object], cache_name: str):
    env = dp.PolicyEnv(demonstration_files=(), prompt_dirs=(PROMPT_DIR,))
    cache = CACHE_DIR / cache_name
    model = dp.CachedModel(dp.openai_model("gpt-4.1-mini"), cache)
    bl = dp.BudgetLimit({dp.NUM_REQUESTS: 1})
    pp = dp.with_budget(bl) @ dp.few_shot(model)
    stream = query.run_toplevel(env, pp)
    res, _ = dp.collect(stream)
    log = list(env.tracer.export_log())
    print(log)
    return res, log


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
