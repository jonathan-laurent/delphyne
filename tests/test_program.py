"""
Testing oracular programs in an end-to-end fashion
"""

from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path
from typing import Any, Literal, cast

import example_strategies as ex
import pytest

import delphyne as dp
from delphyne.utils.yaml import dump_yaml

type APIType = Literal["responses", "chat_completions"]
DEFAULT_TEST_MODEL = "gpt-4.1-mini"

PROMPT_DIR = Path(__file__).parent / "prompts"
DATA_DIR = Path(__file__).parent / "data"
CACHE_DIR = Path(__file__).parent / "cache"
STRATEGY_DIRS = (Path(__file__).parent,)
DEMO_DIR = Path(__file__).parent
STRATEGY_MODULES = ("example_strategies",)


def _make_policy_env(
    *,
    cache: dp.LLMCache | None,
    embeddings_cache: dp.EmbeddingsCache | None,
    demo_files: Sequence[str] = (),
):
    object_loader = dp.ObjectLoader(
        strategy_dirs=STRATEGY_DIRS, modules=STRATEGY_MODULES
    )
    return dp.PolicyEnv(
        demonstration_files=tuple(DEMO_DIR / f for f in demo_files),
        prompt_dirs=(PROMPT_DIR,),
        data_dirs=(DATA_DIR,),
        cache=cache,
        embeddings_cache=embeddings_cache,
        global_embeddings_cache_file=CACHE_DIR / "__embeddings__.h5",
        object_loader=object_loader,
    )


def _log_yaml(log: Sequence[dp.ExportableLogMessage]) -> str:
    return dump_yaml(
        Sequence[dp.ExportableLogMessage], log, exclude_defaults=True
    )


def _log_messages(log: Sequence[dp.ExportableLogMessage]) -> str:
    return "\n".join(e.message for e in log)


def _load_cache(name: str):
    file = CACHE_DIR / (name + ".yaml")
    return dp.load_request_cache(file, mode="read_write")


def _load_embeddings_cache(name: str):
    file = CACHE_DIR / (name + ".embeddings.h5")
    return dp.load_embeddings_cache(file, mode="read_write")


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
    query: dp.Query[Any],
    cache_name: str,
    *,
    budget: int = 1,
    num_completions: int = 1,
    model_name: dp.StandardModelName | str = "gpt-4.1-mini",
    model_options: dp.RequestOptions | None = None,
    model_class: str | None = None,
    api_type: APIType = "chat_completions",
    demo_files: Sequence[str] = (),
    select_examples: dp.ExampleSelector | None = None,
    mode: dp.AnswerMode = None,
):
    # embeddings_cache_name = ...
    if api_type == "responses":
        cache_name = cache_name + "_responses"
    with _load_cache(cache_name) as cache:
        with _load_embeddings_cache(cache_name) as embeddings_cache:
            env = _make_policy_env(
                cache=cache,
                embeddings_cache=embeddings_cache,
                demo_files=demo_files,
            )
            model = dp.standard_model(
                model_name,
                options=model_options,
                model_class=model_class,
                api_type=api_type,
            )
            bl = dp.BudgetLimit({dp.NUM_REQUESTS: budget})
            pp = dp.with_budget(bl) @ dp.few_shot(
                model,
                num_completions=num_completions,
                mode=mode,
                select_examples=select_examples,
            )
            stream = query.run_toplevel(env, pp)
            res, _ = stream.collect()
            log = list(env.tracer.export_log())
            return res, log


def _eval_strategy[N: dp.Node, P, T](
    strategy: dp.StrategyInstance[N, P, T],
    policy: Callable[[dp.LLM], dp.Policy[N, P]],
    cache_name: str,
    max_requests: int = 1,
    max_res: int = 1,
    model_name: dp.StandardModelName | str = DEFAULT_TEST_MODEL,
    api_type: APIType = "chat_completions",
) -> tuple[Sequence[dp.Solution[T]], Sequence[dp.ExportableLogMessage]]:
    if api_type == "responses":
        cache_name = cache_name + "_responses"
    with _load_cache(cache_name) as cache:
        with _load_embeddings_cache(cache_name) as embeddings_cache:
            env = _make_policy_env(
                cache=cache, embeddings_cache=embeddings_cache
            )
            model = dp.standard_model(model_name, api_type=api_type)
            stream = strategy.run_toplevel(env, policy(model))
            budget = dp.BudgetLimit({dp.NUM_REQUESTS: max_requests})
            ret, _spent = stream.collect(budget=budget, num_generated=max_res)
            log = list(env.tracer.export_log())
            return ret, log


@pytest.mark.parametrize("api_type", ["chat_completions", "responses"])
def test_concurrent(api_type: APIType):
    res, _ = _eval_query(
        ex.StructuredOutput(topic="Love"),
        "structured_output_concurrent",
        budget=4,
        num_completions=2,
        api_type=api_type,
    )
    assert len(res) == 8  # 4 requests, 2 completions each time


def test_basic_llm_call():
    with _load_cache("basic_llm_call") as cache:
        env = _make_policy_env(cache=cache, embeddings_cache=None)
        model = dp.openai_model("gpt-4.1-mini")
        pp = dp.few_shot(model)
        bl = dp.BudgetLimit({dp.NUM_REQUESTS: 1})
        policy = dp.take(1) @ dp.with_budget(bl) @ dp.dfs() & ex.MakeSumIP(pp)
        stream = ex.make_sum([1, 2, 3, 4], 7).run_toplevel(env, policy)
        res, _ = stream.collect()
        # print(list(env.tracer.export_log()))
        assert res


@pytest.mark.parametrize("api_type", ["chat_completions", "responses"])
def test_structured_output(api_type: APIType):
    res, _ = _eval_query(
        ex.StructuredOutput(topic="AI"), "structured_output", api_type=api_type
    )
    assert res
    print(res[0].tracked.value)


@pytest.mark.parametrize("api_type", ["chat_completions", "responses"])
def test_propose_article_initial_step(api_type: APIType):
    # Interestingly, the answer content is often empty here despite us
    # explicitly asking for a additional reasoning. Is it because we
    # mandate tool calls? Probably, yes.
    label = "propose_article_initial_step"
    res, _ = _eval_query(
        ex.ProposeArticle(user_name="Alice"), label, api_type=api_type
    )
    v = cast(dp.Response[Any, Any], res[0].tracked.value)
    assert isinstance(v, dp.Response)
    assert isinstance(v.parsed, dp.ToolRequests)
    assert isinstance(v.parsed.tool_calls[0], ex.GetUserFavoriteTopic)


@pytest.mark.parametrize("api_type", ["chat_completions", "responses"])
def test_assistant_priming(api_type: APIType):
    res, log = _eval_query(
        ex.PrimingTest(style="French"), "assistant_priming", api_type=api_type
    )
    assert res
    assert len(log[0].metadata["request"]["chat"]) == 3  # type: ignore


def test_interact():
    with _load_cache("interact") as cache:
        env = _make_policy_env(cache=cache, embeddings_cache=None)
        model = dp.openai_model("gpt-4.1-mini")
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
    query: dp.Query[Any],
    cache_name: str,
    temperature: float = 1.0,
    bias: tuple[str, float] | None = None,
    api_type: APIType = "chat_completions",
):
    if api_type == "responses":
        cache_name = cache_name + "_responses"
    with _load_cache(cache_name) as cache:
        env = _make_policy_env(cache=cache, embeddings_cache=None)
        model = dp.openai_model("gpt-4.1-mini", api_type=api_type)
        bl = dp.BudgetLimit({dp.NUM_REQUESTS: 1})
        pp = dp.with_budget(bl) @ dp.classify(
            model, temperature=temperature, bias=bias
        )
        stream = query.run_toplevel(env, pp)
        res, _ = stream.collect()
        log = list(env.tracer.export_log())
        print(_log_messages(log))
        assert res
        return res[0].meta


@pytest.mark.parametrize("api_type", ["chat_completions", "responses"])
@pytest.mark.parametrize(
    "name, right, wrong",
    [("Jonathan", "common", "rare"), ("X Æ A-Xii:", "rare", "common")],
)
def test_classifiers(name: str, right: str, wrong: str, api_type: APIType):
    res = _eval_classifier_query(
        ex.EvalNameRarity(name),
        f"classify_{right}",
        temperature=1.0,
        api_type=api_type,
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
    print(_log_messages(log))


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
    print(_log_messages(log))
    assert len(res) == 3


#####
##### Embedded Trees and Transformers
#####


def test_embedded_tree_and_transformers():
    res, log = _eval_strategy(
        strategy=ex.recursive_joins(3),
        policy=lambda _: ex.recursive_joins_policy(),
        cache_name="embedded_tree_and_transformers",
    )
    print(_log_messages(log))
    assert res


def test_elim_join():
    res, log = _eval_strategy(
        strategy=ex.recursive_joins(3),
        policy=lambda _: ex.recursive_joins_policy_using_elim_join(),
        cache_name="elim_join",
    )
    print(_log_messages(log))
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
    print(_log_messages(log))
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
    print(_log_messages(log))
    assert res


#####
##### Parralel Policies
#####


@pytest.mark.parametrize("api_type", ["chat_completions", "responses"])
def test_dual_number_parallel_generation(api_type: APIType):
    strategy = ex.dual_number_generation()
    res, _log = _eval_strategy(
        strategy,
        ex.dual_number_generation_parallel_policy,
        cache_name="dual_number_parallel_generation",
        max_requests=10,
        max_res=10,
        api_type=api_type,
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


#####
##### Binarize Values, Iterative Mode for `few_shot`
#####


def test_guess_zero_and_then_one():
    res, _log = _eval_strategy(
        strategy=ex.guess_zero_and_then_one(),
        policy=ex.guess_zero_and_then_one_policy,
        cache_name="guess_zero_and_then_one",
        max_requests=2,
        max_res=1,
        model_name="gpt-5-nano",
    )
    print(_log)
    assert res and res[0].tracked.value == 1


#####
##### Data
#####


def test_strategy_loading_data():
    res, _log = _eval_strategy(
        strategy=ex.strategy_loading_data("yang25"),
        policy=lambda _: ex.strategy_loading_data_policy(),
        cache_name="strategy_loading_data",
        max_requests=0,
    )
    assert res
    print(res[0].tracked.value)


#####
##### Embeddings
#####


def test_similarity_matrix():
    with _load_embeddings_cache("similarity_matrix") as cache:
        env = _make_policy_env(
            cache=None,
            embeddings_cache=cache,
            demo_files=["example_embeddings"],
        )
        sim = env.examples.fetch_example_similarity_matrix(
            "AnswerTriviaQuestion", "text-embedding-3-large"
        )
        assert sim is not None
        assert sim.shape == (5, 5)
        assert (sim.T == sim).all()
        print("\n", sim)


def test_maximum_marginally_relevant():
    with _load_cache("maximum_marginally_relevant") as cache:
        with _load_embeddings_cache(
            "maximum_marginally_relevant"
        ) as embeddings_cache:
            env = _make_policy_env(
                cache=cache,
                embeddings_cache=embeddings_cache,
                demo_files=["example_embeddings"],
            )
            question = "What is the most populated country in Europe?"
            query = ex.AnswerTriviaQuestion(question)
            selector = dp.maximum_marginally_relevant(
                # If setting lambda=1, the two first examples are almost
                # identical
                k=5,
                model_name="text-embedding-3-large",
                lambda_param=0.7,
                always_compute_mmr=True,
            )
            _res = selector(env, query)
            print(_log_yaml(list(env.tracer.export_log())))


def test_example_embeddings():
    question = "What is the most populated country in Europe?"
    res, _log = _eval_query(
        ex.AnswerTriviaQuestion(question),
        "example_embeddings",
        demo_files=["example_embeddings"],
        select_examples=dp.closest_examples(
            k=3, model_name="text-embedding-3-large"
        ),
    )
    # In this case, because we use `closest_examples`, the first two
    # examples are trivial rewordings of the same thing.
    print("\n" + _log_yaml(_log))
    assert res is not None


def test_example_embeddings_empty():
    question = "What is the most populated country in Europe?"
    res, _log = _eval_query(
        ex.AnswerTriviaQuestion(question),
        "example_embeddings_empty",
        demo_files=[],  # No demo file is given
        select_examples=dp.closest_examples(
            k=3, model_name="text-embedding-3-large"
        ),
    )
    print("\n" + _log_yaml(_log))
    assert res is not None


#####
##### Reasoning Cache with Responses API
#####


def _eval_strategy_reasoning[N: dp.Node, P, T](
    strategy: dp.StrategyInstance[N, P, T],
    policy: Callable[[dp.LLM], dp.Policy[N, P]],
    cache_name: str,
    use_reasoning_cache: bool,
    num_requests: int,
    api_type: APIType,
) -> tuple[
    Sequence[dp.Solution[T]], dp.Budget, Sequence[dp.ExportableLogMessage]
]:
    budget = dp.BudgetLimit({dp.NUM_REQUESTS: num_requests})
    if api_type == "responses":
        cache_name = cache_name + "_responses"

    def make_model(use_reasoning_cache: bool):
        return dp.standard_model(
            "gpt-5-nano",
            options={"reasoning_effort": "high"},
            api_type=api_type,
            use_reasoning_cache=use_reasoning_cache,
        )

    with _load_cache(cache_name) as cache:
        env = _make_policy_env(cache=cache, embeddings_cache=None)
        model = make_model(use_reasoning_cache)
        stream = strategy.run_toplevel(env, policy(model))
        log = list(env.tracer.export_log())
        sol, spent = stream.collect(budget=budget, num_generated=1)

    return sol, spent, log


def _test_article_strategy_reasoning_cache(
    strategy: dp.StrategyInstance[dp.Branch, dp.PromptingPolicy, ex.Article],
    num_requests: int = 5,
):
    policy = ex.propose_article_policy
    print("Running with chat completions api")
    _, spent0, log0 = _eval_strategy_reasoning(
        strategy=strategy,
        policy=policy,
        cache_name="interact_no_reasoning_cache",
        use_reasoning_cache=False,
        num_requests=num_requests,
        api_type="chat_completions",
    )
    print("Chat Completions API: ", spent0)

    print("Running with responses api, no reasoning cache")
    _, spent1, log1 = _eval_strategy_reasoning(
        strategy=strategy,
        policy=policy,
        cache_name="interact_no_reasoning_cache",
        use_reasoning_cache=False,
        num_requests=num_requests,
        api_type="responses",
    )
    print("Responses API, no reasoning cache: ", spent1)

    print("Running with responses api and reasoning cache")
    _, spent2, log2 = _eval_strategy_reasoning(
        strategy=strategy,
        policy=policy,
        cache_name="interact_reasoning_cache",
        use_reasoning_cache=True,
        num_requests=num_requests,
        api_type="responses",
    )
    print("Responses with reasoning cache: ", spent2)

    return (spent0, log0), (spent1, log1), (spent2, log2)


def test_reasoning_cache_multi_sequential_tool():
    """
    Instruct LLM to issue tool calls for every user one by one.
    Reasoning can be persisted across multiple tool requests, so we expect
    less output tokens produced when reasoning cache is on.
    There is usually a factor of ~5 savings in this example.
    """
    user_names = ["Alice", "Bob", "Charlie", "Dave", "Eve"]
    strategy = ex.propose_article_multi_user(user_names=user_names)
    print("Test reasoning cache multi sequential tool:")
    _, resp, resp_cache = _test_article_strategy_reasoning_cache(strategy)

    assert all(log.message != "reasoning_cache_miss" for log in resp_cache[1])
    assert resp[0]["num_requests"] == resp_cache[0]["num_requests"]
    assert resp[0]["output_tokens"] > 2 * resp_cache[0]["output_tokens"]
    assert resp[0]["price"] > 2 * resp_cache[0]["price"]

    # Test reasoning cache multi sequential tool:
    # Chat Completions:
    #     'num_requests': 5, 'num_completions': 5,
    #     'input_tokens': 1865, 'output_tokens': 2883,
    #     'cached_input_tokens': 0, 'price': 0.00124645
    # Responses no reasoning cache:
    #     'num_requests': 5, 'num_completions': 5,
    #     'input_tokens': 1510, 'output_tokens': 5838,
    #     'cached_input_tokens': 0, 'price': 0.0024107
    # Responses with reasoning cache:
    #     'num_requests': 5, 'num_completions': 5,
    #     'input_tokens': 5302, 'output_tokens': 1242,
    #     'cached_input_tokens': 1152, 'price': 0.00071006


def test_reasoning_cache_tools_multi_turn():
    """
    Instruct LLM to issue a tool call. When the tool output is processed by
    the LLM, reject the answer and make LLM issue another tool call.
    That means tool calls are interleaved by user messages. Reasoning cache
    does not benefit us in this case, as indicated by OpenAI.
    """
    strategy = ex.propose_article_multi_turn()
    print("Test reasoning cache tools multi turn:")
    _, _, resp_cache = _test_article_strategy_reasoning_cache(strategy)

    assert all(log.message != "reasoning_cache_miss" for log in resp_cache[1])

    # Test reasoning cache tools multi turn:
    # Chat Completions:
    #     'num_requests': 5, 'num_completions': 5,
    #     'input_tokens': 1657, 'output_tokens': 5593,
    #     'cached_input_tokens': 0, 'price': 0.0023200499999999997
    # Responses no reasoning cache:
    #     'num_requests': 5, 'num_completions': 5,
    #     'input_tokens': 1335, 'output_tokens': 6542,
    #     'cached_input_tokens': 0, 'price': 0.00268355
    # Responses with reasoning cache:
    #     'num_requests': 5, 'num_completions': 5,
    #     'input_tokens': 3100, 'output_tokens': 5142,
    #     'cached_input_tokens': 2048, 'price': 0.00211964


def test_reasoning_cache_no_tools_multi_turn():
    """
    Now, there are no tool calls, just a multi-turn conversation.
    Reasoning cache does not benefit us in this case, as indicated by OpenAI.
    """
    strategy = ex.propose_article_no_tool_reject(topic_name="Soccer")
    print("Test reasoning cache no tools multi turn:")
    _, _, resp_cache = _test_article_strategy_reasoning_cache(strategy)

    assert all(log.message != "reasoning_cache_miss" for log in resp_cache[1])

    # Test reasoning cache no tools multi turn:
    # Chat Completions:
    #     'num_requests': 5, 'num_completions': 5,
    #     'input_tokens': 992, 'output_tokens': 8805,
    #     'cached_input_tokens': 0, 'price': 0.0035716
    # Responses no reasoning cache:
    #     'num_requests': 5, 'num_completions': 5,
    #     'input_tokens': 1122, 'output_tokens': 8689,
    #     'cached_input_tokens': 0, 'price': 0.0035317
    # Responses with reasoning cache:
    #     'num_requests': 5, 'num_completions': 5,
    #     'input_tokens': 1130, 'output_tokens': 9666,
    #     'cached_input_tokens': 0, 'price': 0.0039229
