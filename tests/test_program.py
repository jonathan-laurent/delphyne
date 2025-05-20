"""
Testing oracular programs in an end-to-end fashion
"""

from pathlib import Path

import example_strategies as ex

import delphyne as dp

PROMPT_DIR = Path(__file__).parent / "prompts"
CACHE_DIR = Path(__file__).parent / "cache"


def test_basic_llm_call():
    env = dp.PolicyEnv(demonstration_files=(), prompt_dirs=(PROMPT_DIR,))
    cache = CACHE_DIR / "basic_llm_call"
    model = dp.CachedModel(dp.openai_model("gpt-4o"), cache)
    pp = dp.few_shot(model)
    bl = dp.BudgetLimit({dp.NUM_REQUESTS: 1})
    policy = dp.take(1) @ (dp.with_budget(bl) @ dp.dfs()), ex.MakeSumIP(pp)
    stream = ex.make_sum([1, 2, 3, 4], 7).run_toplevel(env, policy)
    res, _ = dp.collect(stream)
    print(list(env.tracer.export_log()))
    assert res
