"""
Test search strategies.
"""

import example_strategies as ex
from test_demo_interpreter import load_demo

import delphyne as dp
import delphyne.stdlib.mock as mock


def _trivial_policy_env() -> dp.PolicyEnv:
    return dp.PolicyEnv(
        object_loader=dp.ObjectLoader.trivial(),
        prompt_dirs=(),
        demonstration_files=(),
        data_dirs=(),
    )


def test_search_synthesis():
    demo = load_demo("synthetize_fun_demo")
    assert isinstance(demo, dp.StrategyDemo)
    env = _trivial_policy_env()
    vars = ["x", "y"]
    prop = (["a", "b"], "F(a, b) == F(b, a) and F(0, 1) == 2")
    pp = mock.demo_mock_oracle(demo)
    inner_policy = ex.SynthetizeFunIP(
        conjecture=dp.dfs() & pp,
        disprove=dp.dfs() & pp,
        aggregate=pp,
    )
    policy = ex.just_guess() & inner_policy
    stream = ex.synthetize_fun(vars, prop).run_toplevel(env, policy)
    res, _ = stream.collect(num_generated=1)
    assert res
    print(res[0].tracked.value)


def test_cached_computations():
    env = _trivial_policy_env()
    # tr = Trans[N, M](dp.elim_compute)
    policy = dp.dfs() @ dp.elim_compute() & None
    stream = ex.test_cached_computations(1).run_toplevel(env, policy)
    res, _ = stream.collect()
    assert res
    print(res[0].tracked.value)


def test_bestfs():
    # 6 requests are necessary to generate the first 4 pairs (2 at the
    # root of the tree and  near the leaves). With one additional
    # request, we generate one more answer.
    REQUESTS_LIMIT = 7

    def oracle(query: object):
        i = 1
        while True:
            yield dp.Answer(None, str(i))
            i += 1

    env = _trivial_policy_env()
    pp = mock.fixed_oracle(oracle)
    policy = ex.generate_pairs_policy(pp)
    stream = ex.generate_pairs().run_toplevel(env, policy)
    budget = dp.BudgetLimit({dp.NUM_REQUESTS: REQUESTS_LIMIT})
    ret, spent = stream.collect(budget=budget)
    res = [x.tracked.value for x in ret]
    assert res == [(1, 1), (2, 1), (2, 2), (1, 2), (2, 3)]
    assert spent[dp.NUM_REQUESTS] == REQUESTS_LIMIT
