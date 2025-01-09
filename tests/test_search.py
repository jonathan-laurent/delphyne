"""
Test search strategies.
"""

import asyncio

import example_strategies as ex
from test_demo_interpreter import load_demo

import delphyne as dp
import delphyne.stdlib.mock as mock


def test_search_synthesis():
    pass
    demo = load_demo("synthetize_fun_demo")
    env = dp.PolicyEnv((), ())  # Won't be used
    vars = ["x", "y"]
    prop = (["a", "b"], "F(a, b) == F(b, a) and F(0, 1) == 2")
    pp = mock.demo_mock_oracle(demo)
    inner_policy = ex.SynthetizeFunIP(
        conjecture=(dp.dfs(), pp),
        disprove=(dp.dfs(), pp),
        aggregate=pp,
    )
    policy = (ex.just_guess(), inner_policy)
    stream = ex.synthetize_fun(vars, prop).run_toplevel(env, policy)
    res = asyncio.run(dp.take_first(stream))
    assert res is not None
    print(res)
