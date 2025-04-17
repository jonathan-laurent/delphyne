"""
Standard commands for running strategies.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import delphyne.analysis as analysis
import delphyne.analysis.feedback as fb
import delphyne.core as dp
import delphyne.stdlib as std
import delphyne.stdlib.tasks as ta

DEFAULT_REFRESH_RATE_IN_SECONDS = 5


@dataclass
class RunStrategyResponse:
    success: bool
    raw_trace: dp.ExportableTrace
    log: Sequence[dp.ExportableLogMessage]
    browsable_trace: fb.Trace


@dataclass
class RunLoadedStrategyArgs[N: dp.Node, P, T]:
    strategy: dp.StrategyComp[N, P, T]
    policy: dp.Policy[N, P]
    num_generated: int = 1
    budget: dict[str, float] | None = None


async def run_loaded_strategy[N: dp.Node, P, T](
    task: ta.TaskContext[ta.CommandResult[RunStrategyResponse]],
    exe: ta.CommandExecutionContext,
    args: RunLoadedStrategyArgs[N, P, T],
):
    # TODO: respect refresh rate
    refresh_rate = exe.refresh_rate
    if refresh_rate is None:
        refresh_rate = DEFAULT_REFRESH_RATE_IN_SECONDS
    env = dp.PolicyEnv(
        exe.base.strategy_dirs,
        exe.demo_files,
        do_not_match_identical_queries=True,
    )
    cache: dp.TreeCache = {}
    monitor = dp.TreeMonitor(cache, hooks=[dp.tracer_hook(env.tracer)])
    tree = dp.reify(args.strategy, monitor)
    search_policy, inner_policy = args.policy
    stream = search_policy(tree, env, inner_policy)
    if args.budget is not None:
        stream = std.stream_with_budget(stream, dp.BudgetLimit(args.budget))
    stream = std.stream_take(stream, 1)
    success = False
    total_budget = dp.Budget.zero()

    def compute_result():
        trace = env.tracer.trace
        raw_trace = trace.export()
        browsable_trace = analysis.compute_browsable_trace(trace, cache)
        log = list(env.tracer.export_log())
        response = RunStrategyResponse(
            success, raw_trace, log, browsable_trace
        )
        return ta.CommandResult([], response)

    def compute_status():
        num_nodes = len(env.tracer.trace.nodes)
        num_requests = total_budget.values.get(std.NUM_REQUESTS)
        if num_requests is not None:
            return f"{num_nodes} nodes, {num_requests} requests"
        else:
            return f"{num_nodes} nodes"

    last_refreshed = time.time()
    async for msg in stream:
        match msg:
            case dp.Yield():
                success = True
            case dp.Spent(b):
                total_budget += b
            case dp.Barrier():
                pass
        if time.time() - last_refreshed > refresh_rate:
            await task.set_result(compute_result())
            last_refreshed = time.time()
        await task.set_status(compute_status())
    await task.set_result(compute_result())


@dataclass
class RunStrategyArgs:
    strategy: str
    args: dict[str, object]
    policy: str
    policy_args: dict[str, object]
    num_generated: int
    budget: dict[str, float]


async def run_strategy(
    task: ta.TaskContext[ta.CommandResult[RunStrategyResponse]],
    exe: ta.CommandExecutionContext,
    args: RunStrategyArgs,
):
    loader = analysis.ObjectLoader(exe.base)
    strategy = loader.load_strategy_instance(args.strategy, args.args)
    policy = loader.load_and_call_function(args.policy, args.policy_args)
    await run_loaded_strategy(
        task,
        exe,
        RunLoadedStrategyArgs(
            strategy, policy, args.num_generated, args.budget
        ),
    )
