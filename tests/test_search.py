"""
Test search strategies.
"""

import asyncio
from pathlib import Path

import pytest
from test_server import load_demo
from test_strategies import (
    ConjectureExpr,
    ProposeCex,
    RemoveDuplicates,
    generate_pairs,
    synthetize_fun,
)

from delphyne.core.strategies import StrategyTree
from delphyne.server.basic_launcher import BasicLauncher
from delphyne.server.commands import (
    BudgetLimit,
    CommandExecutionContext,
    CommandResult,
    RunStrategyCmd,
    run_strategy,
)
from delphyne.server.evaluate_demo import ExecutionContext
from delphyne.stdlib.generators import Budget, BudgetCounter, GenEnv, GenRet
from delphyne.stdlib.mock_oracles import DemoMockedSearchEnv, MockedSearchEnv
from delphyne.stdlib.search_envs import Params


async def run_to_success[T](
    gen: GenRet[T], max_attempts: int | None = None
) -> T | None:  # fmt: skip
    i = 0
    async for resp in gen:
        if resp.items:
            return resp.items[0].value
        if max_attempts is not None and i >= max_attempts:
            return None
        i += 1


def test_search_synthesis():
    query_classes = [ConjectureExpr, ProposeCex, RemoveDuplicates]
    demo = load_demo("synthetize_fun_demo")
    demo_paths: list[Path] = []
    strategy_dirs: list[Path] = [Path(__file__).parent]
    exe_context = ExecutionContext(strategy_dirs, [])
    env = DemoMockedSearchEnv(demo_paths, exe_context, demo, query_classes)
    param = Params(env)
    vars = ["x", "y"]
    prop = (["a", "b"], "F(a, b) == F(b, a) and F(0, 1) == 2")
    tree = StrategyTree.new(synthetize_fun(vars, prop))
    search = synthetize_fun.search_policy
    assert search is not None
    gen = search(GenEnv([], "lazy"), tree, param)
    res = asyncio.run(run_to_success(gen, max_attempts=10))
    assert res is not None
    print(res)


def test_bfs():

    # 6 requests are necessary to generate the first 4 pairs (2 at the
    # root of the tree and  near the leaves). With one additional
    # request, we generate one more answer.
    REQUESTS_LIMIT = 7
    ATTEMPTS_LIMIT = 10

    def oracle(query: object, prompt: object):
        i = 1
        while True:
            yield str(i)
            i += 1

    counter = BudgetCounter(Budget.limit(num_requests=REQUESTS_LIMIT))
    gen_env = GenEnv([counter], "lazy")
    search_env = MockedSearchEnv(oracle)
    params = Params(search_env)
    tree = StrategyTree.new(generate_pairs())
    search = generate_pairs.search_policy
    assert search is not None
    gen = search(gen_env, tree, params)

    async def explore():
        i = 0
        acc: list[tuple[int, int]] = []
        async for resp in gen:
            for item in resp.items:
                acc.append(item.value)
            # print(resp)
            i += 1
            if not gen_env.budget_left() or i > ATTEMPTS_LIMIT:
                break
        return acc

    generated = asyncio.run(explore())
    assert generated == [(1, 1), (2, 1), (2, 2), (1, 2), (1, 3)]


@pytest.mark.skip(reason="Uses OpenAI credits")
def test_run_strategy_command():
    dir = Path(__file__).parent
    exe = CommandExecutionContext(
        ExecutionContext(
            strategy_dirs=[dir], modules=["test_strategies", "test_prompts"]
        ),
        demo_files=[dir / "examples.demo.yaml"],
    )
    cmd = RunStrategyCmd(
        strategy="synthetize_fun",
        args={
            "vars": ["x", "y"],
            "prop": (["a"], "F(a, a) == 0 and F(0, 0) == 0 and F(1, 0) > 0"),
        },
        params={},
        budget=BudgetLimit(num_requests=1),
    )
    launcher = BasicLauncher()

    async def print_output():
        async for s in launcher(None, CommandResult, run_strategy, exe, cmd):
            print(s)

    asyncio.run(print_output())


if __name__ == "__main__":
    # test_search_synthesis()
    # test_run_strategy_command()
    test_bfs()
