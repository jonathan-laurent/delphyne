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


def _load_cache(name: str):
    file = CACHE_DIR / (name + ".yaml")
    return dp.load_request_cache(file, mode="read_write")


def test_query_properties():
    q1 = ex.StructuredOutput(topic="AI")
    assert q1.query_settings(None).structured_output
    q2 = ex.MakeSum([1, 2, 3, 4], 7)
    assert not q2.query_settings(None).structured_output
    q3 = ex.ProposeArticle(user_name="Alice")
    assert (tools := q3.query_settings(None).tools) is not None
    assert tools.force_tool_call
    assert len(tools.tool_types) == 3  # Counting the answer tool


def _eval_query(
    query: dp.Query[object],
    cache_name: str,
    budget: int = 1,
    num_completions: int = 1,
    model_name: dp.StandardModelName | str = "gpt-4.1-mini",
    model_options: dp.RequestOptions | None = None,
    model_class: str | None = None,
    mode: dp.AnswerMode = None,
):
    env = dp.PolicyEnv(
        demonstration_files=(), prompt_dirs=(PROMPT_DIR,), data_dirs=()
    )
    with _load_cache(cache_name) as cache:
        base_model = dp.standard_model(
            model_name, options=model_options, model_class=model_class
        )
        model = dp.CachedModel(base_model, cache)
        bl = dp.BudgetLimit({dp.NUM_REQUESTS: budget})
        pp = dp.with_budget(bl) @ dp.few_shot(
            model, num_completions=num_completions, mode=mode
        )
        stream = query.run_toplevel(env, pp)
        res, _ = stream.collect()
        log = list(env.tracer.export_log())
        print(log)
        return res, log


def _eval_strategy[N: dp.Node, P, T](
    strategy: dp.StrategyInstance[N, P, T],
    policy: Callable[[dp.LLM], dp.Policy[N, P]],
    cache_name: str,
    max_requests: int = 1,
    max_res: int = 1,
    model_name: dp.StandardModelName | str = "gpt-4.1-mini",
) -> tuple[Sequence[dp.Solution[T]], str]:
    env = dp.PolicyEnv(
        prompt_dirs=[PROMPT_DIR], demonstration_files=(), data_dirs=()
    )
    with _load_cache(cache_name) as cache:
        model = dp.CachedModel(dp.standard_model(model_name), cache)
        stream = strategy.run_toplevel(env, policy(model))
        budget = dp.BudgetLimit({dp.NUM_REQUESTS: max_requests})
        ret, _spent = stream.collect(budget=budget, num_generated=max_res)
        log = list(env.tracer.export_log())
        log_str = "\n".join(e.message for e in log)
        return ret, log_str


def test_concurrent():
    res, _ = _eval_query(
        ex.StructuredOutput(topic="Love"),
        "structured_output_concurrent",
        budget=4,
        num_completions=2,
    )
    assert len(res) == 8  # 4 requests, 2 completions each time


def test_basic_llm_call():
    env = dp.PolicyEnv(
        demonstration_files=(), prompt_dirs=(PROMPT_DIR,), data_dirs=()
    )
    with _load_cache("basic_llm_call") as cache:
        model = dp.CachedModel(dp.openai_model("gpt-4.1-mini"), cache)
        pp = dp.few_shot(model)
        bl = dp.BudgetLimit({dp.NUM_REQUESTS: 1})
        policy = dp.take(1) @ dp.with_budget(bl) @ dp.dfs() & ex.MakeSumIP(pp)
        stream = ex.make_sum([1, 2, 3, 4], 7).run_toplevel(env, policy)
        res, _ = stream.collect()
        # print(list(env.tracer.export_log()))
        assert res


def test_structured_output():
    res, _ = _eval_query(ex.StructuredOutput(topic="AI"), "structured_output")
    assert res
    print(res[0].tracked.value)


def test_propose_article_initial_step():
    # Interestingly, the answer content is often empty here despite us
    # explicitly asking for a additional reasoning. Is it because we
    # mandate tool calls? Probably, yes.
    label = "propose_article_initial_step"
    res, _ = _eval_query(ex.ProposeArticle(user_name="Alice"), label)
    v = cast(dp.Response[Any, Any], res[0].tracked.value)
    assert isinstance(v, dp.Response)
    assert isinstance(v.parsed, dp.ToolRequests)
    assert isinstance(v.parsed.tool_calls[0], ex.GetUserFavoriteTopic)


def test_assistant_priming():
    res, log = _eval_query(ex.PrimingTest(style="French"), "assistant_priming")
    assert res
    assert len(log[0].metadata["request"]["chat"]) == 3  # type: ignore


def test_interact():
    env = dp.PolicyEnv(
        demonstration_files=(), prompt_dirs=(PROMPT_DIR,), data_dirs=()
    )
    with _load_cache("interact") as cache:
        model = dp.CachedModel(dp.openai_model("gpt-4.1-mini"), cache)
        pp = dp.few_shot(model)
        bl = dp.BudgetLimit({dp.NUM_REQUESTS: 2})
        policy = dp.take(1) @ dp.with_budget(bl) @ dp.dfs() & pp
        stream = ex.propose_article("Jonathan").run_toplevel(env, policy)
        res, _ = stream.collect()
        print(list(env.tracer.export_log()))
        assert res


def test_both_tool_call_and_structured_output():
    strategy = ex.propose_article_structured("Jonathan")
    policy = ex.propose_article_policy
    cache_name = "both_tool_call_and_structured_output"
    res, _log = _eval_strategy(
        strategy, policy, cache_name, max_requests=2, max_res=1
    )
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
    temperature: float = 1.0,
    bias: tuple[str, float] | None = None,
):
    env = dp.PolicyEnv(
        demonstration_files=(), prompt_dirs=(PROMPT_DIR,), data_dirs=()
    )
    with _load_cache(cache_name) as cache:
        model = dp.CachedModel(dp.openai_model("gpt-4.1-mini"), cache)
        bl = dp.BudgetLimit({dp.NUM_REQUESTS: 1})
        pp = dp.with_budget(bl) @ dp.classify(
            model, temperature=temperature, bias=bias
        )
        stream = query.run_toplevel(env, pp)
        res, _ = stream.collect()
        log = list(env.tracer.export_log())
        print(log)
        assert res
        return res[0].meta


@pytest.mark.parametrize(
    "name, right, wrong",
    [("Jonathan", "common", "rare"), ("X Æ A-Xii:", "rare", "common")],
)
def test_classifiers(name: str, right: str, wrong: str):
    res = _eval_classifier_query(
        ex.EvalNameRarity(name), f"classify_{right}", temperature=1.0
    )
    assert isinstance(res, dp.ProbInfo)
    D = {k.value: v for k, v in res.distr}
    print(D)
    assert D[right] > D[wrong]


def test_apply_bias():
    # Test generated by Claude Sonnet.
    from delphyne.stdlib.queries import _apply_bias  # type: ignore

    # Test basic bias application
    original_probs = {"A": 0.3, "B": 0.7}
    bias = ("A", 0.5)
    result = _apply_bias(original_probs, bias)

    # Expected: (1-0.5) * {A: 0.3, B: 0.7} + 0.5 * {A: 1, B: 0}
    # = {A: 0.15, B: 0.35} + {A: 0.5, B: 0}
    # = {A: 0.65, B: 0.35}
    expected = {"A": 0.65, "B": 0.35}
    assert abs(result["A"] - expected["A"]) < 1e-10
    assert abs(result["B"] - expected["B"]) < 1e-10

    # Test bias toward element not in original distribution
    original_probs2 = {"X": 0.4, "Y": 0.6}
    bias2 = ("Z", 0.3)
    result2 = _apply_bias(original_probs2, bias2)

    # Expected: (1-0.3) * {X: 0.4, Y: 0.6, Z: 0} + 0.3 * {X: 0, Y: 0, Z: 1}
    # = {X: 0.28, Y: 0.42, Z: 0} + {X: 0, Y: 0, Z: 0.3}
    # = {X: 0.28, Y: 0.42, Z: 0.3}
    expected2 = {"X": 0.28, "Y": 0.42, "Z": 0.3}
    assert abs(result2["X"] - expected2["X"]) < 1e-10
    assert abs(result2["Y"] - expected2["Y"]) < 1e-10
    assert abs(result2["Z"] - expected2["Z"]) < 1e-10

    # Test extreme bias (p=1)
    original_probs3 = {"P": 0.8, "Q": 0.2}
    bias3 = ("P", 1.0)
    result3 = _apply_bias(original_probs3, bias3)

    # Expected: (1-1) * {P: 0.8, Q: 0.2} + 1 * {P: 1, Q: 0}
    # = {P: 0, Q: 0} + {P: 1, Q: 0}
    # = {P: 1, Q: 0}
    expected3 = {"P": 1.0, "Q": 0.0}
    assert abs(result3["P"] - expected3["P"]) < 1e-10
    assert abs(result3["Q"] - expected3["Q"]) < 1e-10


#####
##### Diverse Providers
#####


@pytest.mark.parametrize(
    "model,options,options_label",
    [
        ("mistral-small-2503", None, None),
        ("gemini-2.5-flash", None, None),
        ("gpt-5-mini", {"reasoning_effort": "minimal"}, "minimal_reasoning"),
        ("gpt-5-mini", {"reasoning_effort": "high"}, "high_reasoning"),
        ("deepseek-chat", None, None),
        # ("deepseek-reasoner", None, None),  # long latency
    ],
)
def test_provider(
    model: dp.StandardModelName | str,
    options: dp.RequestOptions | None,
    options_label: str | None,
):
    cache_name = f"provider_{model}"
    if options_label is not None:
        cache_name += f"_{options_label}"
    query = ex.MakeSum(allowed=[1, 2, 3, 4], goal=5)
    res, _ = _eval_query(
        query,
        cache_name,
        model_name=model,
        model_options=options,
        model_class=options_label,  # for testing model classes
    )
    assert res
    print(res[0].tracked.value)


@pytest.mark.parametrize(
    "model",
    [
        "mistral-small-2503",
        # "deepseek-chat",  # TODO" test structured output with DeepSeek
    ],
)
def test_structured_output_provider(model: dp.StandardModelName | str):
    cache_name = f"provider_structured_{model}"
    query = ex.StructuredOutput(topic="AI")
    _ = _eval_query(query, cache_name, model_name=model)
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
    policy = partial(ex.obtain_item_policy, num_completions=1)
    cache_name = "abduction"
    res, log = _eval_strategy(
        strategy, policy, cache_name, max_requests=10, max_res=1
    )
    # assert res
    print(res)
    print()
    print(log)


#####
##### Sequencing
#####


def test_sequence():
    strategy = ex.make_sum([1, 2, 3, 4], 7)

    def policy(
        model: dp.LLM,
    ):
        one_req = dp.with_budget(dp.BudgetLimit({dp.NUM_REQUESTS: 1}))
        return dp.sequence(
            [
                one_req @ dp.dfs()
                & ex.MakeSumIP(dp.few_shot(model, num_completions=k))
                for k in [1, 2]
            ]
        )

    res, log = _eval_strategy(
        strategy, policy, cache_name="sequence", max_requests=10, max_res=10
    )
    print(log)
    assert len(res) == 3


#####
##### Embedded Trees and Transformers
#####


def test_embedded_tree_and_transformers():
    strategy = ex.recursive_joins(3)

    def policy(_: dp.LLM):
        return ex.recursive_joins_policy()

    res, log = _eval_strategy(
        strategy, policy, cache_name="no_need_for_caching", max_res=1
    )
    print(log)
    assert res


#####
##### DictIPs
#####


def test_make_sum_dict_ip():
    strategy = ex.make_sum_dict_ip([1, 2, 3, 4], 7)
    res, log = _eval_strategy(
        strategy,
        ex.make_sum_dict_ip_policy,
        cache_name="make_sum_dict_ip",
        max_requests=2,
        max_res=1,
    )
    print(log)
    assert res


@pytest.mark.parametrize("shared", [True, False])
def test_dual_number_generation(shared: bool):
    strategy = ex.dual_number_generation()
    res, log = _eval_strategy(
        strategy,
        partial(ex.dual_number_generation_policy, shared=shared),
        cache_name="dual_number_generation",
        max_requests=4,
        max_res=1,
    )
    print(log)
    assert res


#####
##### Parralel Policies
#####


def test_dual_number_parallel_generation():
    strategy = ex.dual_number_generation()
    res, _log = _eval_strategy(
        strategy,
        ex.dual_number_generation_parallel_policy,
        cache_name="dual_number_parallel_generation",
        max_requests=10,
        max_res=10,
    )
    assert res
    assert len(res) == 4


#####
##### Modes and Wrapped Errors
#####


@pytest.mark.parametrize("mode", ["cot", "direct"])
def test_mode_dict(mode: str):
    res, _ = _eval_query(
        ex.GetFavoriteDish(user="Jonathan Laurent"),
        f"mode_dict_{mode}",
        mode=mode,
    )
    assert res


@pytest.mark.parametrize("no_wrap", [True, False])
def test_wrapped_parse_error(no_wrap: bool):
    res, _ = _eval_strategy(
        ex.get_magic_number(),
        partial(ex.get_magic_number_policy, no_wrap=no_wrap),
        f"wrapped_parse_error_{no_wrap}",
        max_requests=2,
    )
    if no_wrap:
        assert not res
    else:
        assert res


#####
##### Univeresal Queries
#####


def test_make_sum_using_guess():
    strategy = ex.make_sum_using_guess(allowed=[1, 2, 3, 4], goal=5)
    res, _log = _eval_strategy(
        strategy,
        ex.make_sum_using_guess_policy,
        cache_name="make_sum_using_guess",
        max_requests=1,
        max_res=1,
        model_name="gpt-5-mini",
    )
    assert res
