"""
Testing oracular programs in an end-to-end fashion
"""

from pathlib import Path

import example_strategies as ex
from why3py import dataclass

import delphyne as dp

#####
##### Strategies
#####


@dataclass
class Article:
    title: str
    authors: list[str]


@dataclass
class StructuredOutput(dp.Query[Article]):
    """
    Generate an article on the given topic. Answer as a JSON object.
    """

    topic: str


#####
##### Main Script
#####


PROMPT_DIR = Path(__file__).parent / "prompts"
CACHE_DIR = Path(__file__).parent / "cache"


def test_basic_llm_call():
    env = dp.PolicyEnv(demonstration_files=(), prompt_dirs=(PROMPT_DIR,))
    cache = CACHE_DIR / "basic_llm_call"
    model = dp.CachedModel(dp.openai_model("gpt-4.1-mini"), cache)
    pp = dp.few_shot(model)
    bl = dp.BudgetLimit({dp.NUM_REQUESTS: 1})
    policy = dp.take(1) @ (dp.with_budget(bl) @ dp.dfs()), ex.MakeSumIP(pp)
    stream = ex.make_sum([1, 2, 3, 4], 7).run_toplevel(env, policy)
    res, _ = dp.collect(stream)
    print(list(env.tracer.export_log()))
    assert res


def test_query_properties():
    assert StructuredOutput(topic="AI").query_config().force_structured_output
    q2 = ex.MakeSum([1, 2, 3, 4], 7)
    assert not q2.query_config().force_structured_output


def test_structured_output():
    env = dp.PolicyEnv(demonstration_files=(), prompt_dirs=(PROMPT_DIR,))
    cache = CACHE_DIR / "structured_output"
    model = dp.CachedModel(dp.openai_model("gpt-4.1-mini"), cache)
    bl = dp.BudgetLimit({dp.NUM_REQUESTS: 1})
    pp = dp.with_budget(bl) @ dp.few_shot(model)
    stream = StructuredOutput(topic="AI").run_toplevel(env, pp)
    res, _ = dp.collect(stream)
    # print(list(env.tracer.export_log()))
    assert res
    print(res[0].value)
