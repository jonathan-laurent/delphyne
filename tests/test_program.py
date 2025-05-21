"""
Testing oracular programs in an end-to-end fashion
"""

from pathlib import Path
from typing import Any, ClassVar, cast

import example_strategies as ex
from why3py import dataclass

import delphyne as dp

#####
##### Strategies and Queries
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
    dp.Query[Article | dp.FollowUpRequest[GetUserFavoriteTopic | Calculator]]
):
    user_name: str

    __parser__: ClassVar[dp.ParserSpec] = "final_tool_call"

    __system_prompt__: ClassVar[str] = """
        Find the user's tastes and propose an article for them.
        Provide your final answer by calling the `Article` tool.
        Please carefully explain your reasoning before calling any tool.
        """


@dataclass
class PrimingTest(dp.Query[list[str]]):
    """
    Generate a list of nice baby names in the given style.

    End your answer with a triple-backquoted code block containing a
    list of strings as a YAML object.
    """

    style: str

    __parser__: ClassVar[dp.ParserSpec] = dp.yaml_from_last_block

    __instance_prompt__: ClassVar[str] = """
    Style: {{query.style}}
    !<assistant>
    Here are 4 baby names in this style:
    """


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
    # print(list(env.tracer.export_log()))
    assert res


def test_query_properties():
    assert StructuredOutput(topic="AI").query_config().force_structured_output
    q2 = ex.MakeSum([1, 2, 3, 4], 7)
    assert not q2.query_config().force_structured_output
    q3 = ProposeArticle(user_name="Alice")
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


def test_structured_output():
    res, _ = _eval_query(StructuredOutput(topic="AI"), "structured_output")
    assert res
    print(res[0].value)


def test_propose_article_initial_step():
    # Interestingly, the answer content is often empty here despite us
    # explicitly asking for a additional reasoning. Is it because we
    # mandate tool calls? Probably, yes.
    res, _ = _eval_query(
        ProposeArticle(user_name="Alice"), "propose_article_initial_step"
    )
    v = cast(dp.FollowUpRequest[Any], res[0].value)
    assert isinstance(v, dp.FollowUpRequest)
    assert isinstance(v.tool_calls[0], GetUserFavoriteTopic)


def test_assistant_priming():
    res, log = _eval_query(PrimingTest(style="French"), "assistant_priming")
    assert res
    assert len(log[0].metadata["request"]["chat"]) == 3  # type: ignore
