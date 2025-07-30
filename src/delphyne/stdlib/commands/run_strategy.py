"""
Standard commands for running strategies.
"""

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import delphyne.analysis as analysis
import delphyne.analysis.feedback as fb
import delphyne.core as dp
import delphyne.stdlib as std
import delphyne.stdlib.tasks as ta
from delphyne.utils.typing import pydantic_dump


@dataclass
class RunStrategyResponse:
    success: bool
    values: Sequence[Any | None]
    spent_budget: Mapping[str, float]
    raw_trace: dp.ExportableTrace | None
    log: Sequence[dp.ExportableLogMessage] | None
    browsable_trace: fb.Trace | None


@dataclass
class RunLoadedStrategyArgs[N: dp.Node, P, T]:
    strategy: dp.StrategyComp[N, P, T]
    policy: dp.Policy[N, P]
    num_generated: int = 1
    budget: dict[str, float] | None = None
    cache_dir: str | None = None
    cache_mode: dp.CacheMode = "read_write"
    cache_format: dp.CacheFormat = "yaml"
    export_raw_trace: bool = True
    export_log: bool = True
    export_browsable_trace: bool = True


async def run_loaded_strategy[N: dp.Node, P, T](
    task: ta.TaskContext[ta.CommandResult[RunStrategyResponse]],
    exe: ta.CommandExecutionContext,
    args: RunLoadedStrategyArgs[N, P, T],
):
    cache_dir = None
    if args.cache_dir is not None:
        assert exe.cache_root is not None, "Nonspecified cache root."
        cache_dir = exe.cache_root / args.cache_dir
    env = dp.PolicyEnv(
        prompt_dirs=exe.prompt_dirs,
        data_dirs=exe.data_dirs,
        demonstration_files=exe.demo_files,
        cache_dir=cache_dir,
        cache_mode=args.cache_mode,
        cache_format=args.cache_format,
        do_not_match_identical_queries=True,
        make_cache=std.LLMCache,
    )
    cache: dp.TreeCache = {}
    monitor = dp.TreeMonitor(cache, hooks=[dp.tracer_hook(env.tracer)])
    tree = dp.reify(args.strategy, monitor)
    search_policy, inner_policy = args.policy
    stream = search_policy(tree, env, inner_policy)
    if args.budget is not None:
        stream = std.stream_with_budget(stream, dp.BudgetLimit(args.budget))
    stream = std.stream_take(stream, args.num_generated)
    results: list[T] = []
    success = False
    total_budget = dp.Budget.zero()

    def serialize_result(res: T) -> Any | None:
        ret_type = args.strategy.return_type()
        return pydantic_dump(ret_type, res)

    def compute_result():
        trace = env.tracer.trace
        raw_trace = trace.export() if args.export_raw_trace else None
        browsable_trace = (
            analysis.compute_browsable_trace(trace, cache)
            if args.export_browsable_trace
            else None
        )
        log = list(env.tracer.export_log()) if args.export_log else None
        values = [serialize_result(r) for r in results]
        response = RunStrategyResponse(
            success,
            values,
            total_budget.values,
            raw_trace,
            log,
            browsable_trace,
        )
        return ta.CommandResult([], response)

    def compute_status():
        num_nodes = len(env.tracer.trace.nodes)
        num_requests = total_budget.values.get(std.NUM_REQUESTS)
        if num_requests is not None:
            # If num_requests is a float equal to an int, cast to int
            # for display
            if isinstance(num_requests, float) and num_requests.is_integer():
                num_requests = int(num_requests)
            return f"{num_nodes} nodes, {num_requests} requests"
        else:
            return f"{num_nodes} nodes"

    last_refreshed_result = time.time()
    last_refreshed_status = time.time()
    # TODO: generating each element is blocking here. Should we spawn a
    # thread for every new element?
    for msg in stream:
        match msg:
            case dp.Yield():
                success = True
                results.append(msg.value.value)
            case dp.Spent(b):
                total_budget += b
            case dp.Barrier():
                pass
        if (
            exe.result_refresh_period is not None
            and time.time() - last_refreshed_result > exe.result_refresh_period
        ):
            await task.set_result(compute_result())
            last_refreshed_result = time.time()
        if (
            exe.status_refresh_period is not None
            and time.time() - last_refreshed_status > exe.status_refresh_period
        ):
            await task.set_status(compute_status())
            last_refreshed_status = time.time()
    await task.set_result(compute_result())


@dataclass
class RunStrategyArgs:
    strategy: str
    args: dict[str, object]
    policy: str
    policy_args: dict[str, object]
    num_generated: int
    budget: dict[str, float]
    cache_dir: str | None = None
    cache_mode: dp.CacheMode = "read_write"
    cache_format: dp.CacheFormat = "yaml"
    export_raw_trace: bool = True
    export_log: bool = True
    export_browsable_trace: bool = True


async def run_strategy(
    task: ta.TaskContext[ta.CommandResult[RunStrategyResponse]],
    exe: ta.CommandExecutionContext,
    args: RunStrategyArgs,
):
    loader = analysis.ObjectLoader(exe.base)
    strategy = loader.load_strategy_instance(args.strategy, args.args)
    policy = loader.load_and_call_function(args.policy, args.policy_args)
    await run_loaded_strategy(
        task=task,
        exe=exe,
        args=RunLoadedStrategyArgs(
            strategy=strategy,
            policy=policy,
            num_generated=args.num_generated,
            budget=args.budget,
            cache_dir=args.cache_dir,
            cache_mode=args.cache_mode,
            cache_format=args.cache_format,
            export_raw_trace=args.export_raw_trace,
            export_log=args.export_log,
            export_browsable_trace=args.export_browsable_trace,
        ),
    )
