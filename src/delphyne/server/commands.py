"""
Custom Delphyne Commands
"""

import asyncio
import traceback
from dataclasses import dataclass
from inspect import isclass
from pathlib import Path
from typing import Any, cast

import delphyne.analysis as analysis
import delphyne.analysis.feedback as fb
import delphyne.core as dp
import delphyne.server.tasks as ta
import delphyne.stdlib as std
import delphyne.stdlib.models as mo
import delphyne.stdlib.queries as qu
import delphyne.utils.typing as ty

DEFAULT_OPENAI_MODEL = "gpt-4o"


@dataclass
class CommandExecutionContext:
    base: analysis.DemoExecutionContext
    demo_files: list[Path]


@dataclass
class CommandSpec:
    command: str
    args: dict[str, object]


@dataclass
class CommandResult[T]:
    diagnostics: list[fb.Diagnostic]
    result: T


async def execute_command(
    task: ta.TaskContext[CommandResult[Any]],
    exe: CommandExecutionContext,
    cmd: CommandSpec,
):
    try:
        match cmd.command:
            case "test_command":
                await test_command(task)
            case "answer_query":
                payload = ty.pydantic_load(AnswerQueryCmd, cmd.args)
                await answer_query(task, exe, payload)
            case "run_strategy":
                payload = ty.pydantic_load(RunStrategyCmd, cmd.args)
                await run_strategy(task, exe, payload)
            case _:
                error = ("error", f"Unknown command: {cmd.command}")
                await task.set_result(CommandResult([error], None))
    except Exception as e:
        error = (
            "error",
            f"Internal error: {repr(e)}\n\n{traceback.format_exc()}",
        )
        await task.set_result(CommandResult([error], None))


async def test_command(task: ta.TaskContext[CommandResult[list[int]]]):
    await task.set_status("counting...")
    for i in range(11):
        await asyncio.sleep(0.5)
        await task.set_result(CommandResult([], list(range(i + 1))))
        print(i)
    await task.set_status("done")
    await asyncio.sleep(0.1)
    print("done")


#####
##### Answer Queries
#####


@dataclass
class AnswerQueryCmd:
    query: str
    completions: int
    prompt_only: bool
    params: dict[str, object]
    options: mo.RequestOptions
    args: dict[str, object]


@dataclass
class AnswerQueryResponse:
    prompt: mo.Chat | None
    response: str | None


async def answer_query(
    task: ta.TaskContext[CommandResult[AnswerQueryResponse]],
    exe: CommandExecutionContext,
    cmd: AnswerQueryCmd,
):
    # TODO: no examples for now. Also, we have to externalize this anyway.
    loader = analysis.ObjectLoader(exe.base)
    query = loader.load_query(cmd.query, cmd.args)
    env = dp.TemplatesManager(exe.base.strategy_dirs)
    prompt = qu.create_prompt(query, [], cmd.params, env)
    await task.set_result(CommandResult([], AnswerQueryResponse(prompt, None)))
    if cmd.prompt_only:
        return
    answer = ""
    model = std.openai_model(DEFAULT_OPENAI_MODEL)
    async for chunk in model.stream_request(prompt, cmd.options):
        answer += chunk
        await task.set_result(
            CommandResult([], AnswerQueryResponse(prompt, answer))
        )


#####
##### Run Search
#####


@dataclass
class RunStrategyCmd:
    strategy: str
    args: dict[str, object]
    policy: str
    policy_args: dict[str, object]
    budget: dict[str, float]


@dataclass
class RunStrategyResponse:
    # TODO: export the log too.
    success: bool
    raw_trace: dp.ExportableTrace
    browsable_trace: fb.Trace


async def run_strategy(
    task: ta.TaskContext[CommandResult[RunStrategyResponse]],
    exe: CommandExecutionContext,
    cmd: RunStrategyCmd,
):
    loader = analysis.ObjectLoader(exe.base)
    strategy = loader.load_strategy_instance(cmd.strategy, cmd.args)
    env = dp.PolicyEnv(exe.base.strategy_dirs, exe.demo_files)
    cache: dp.TreeCache = {}
    monitor = dp.TreeMonitor(cache, hooks=[dp.tracer_hook(env.tracer)])
    tree = dp.reify(strategy, monitor)
    policy = loader.load_and_call_function(cmd.policy, cmd.policy_args)
    assert isinstance(policy, tuple)
    policy = cast(dp.Policy[Any, Any], policy)
    assert len(policy) == 2
    search_policy, inner_policy = policy
    if not isinstance(ipt := strategy.inner_policy_type(), ty.NoTypeInfo):
        if isclass(ipt):
            assert isinstance(inner_policy, ipt)
    stream = search_policy(tree, env, inner_policy)
    budget = dp.BudgetLimit(cmd.budget)
    stream = std.take(1)(std.with_budget(budget)(stream))
    success = False
    total_budget = dp.Budget.zero()

    def compute_result():
        trace = env.tracer.trace
        raw_trace = trace.export()
        browsable_trace = analysis.compute_browsable_trace(trace, cache)
        response = RunStrategyResponse(success, raw_trace, browsable_trace)
        return CommandResult([], response)

    def compute_status():
        num_nodes = len(env.tracer.trace.nodes)
        num_requests = total_budget.values.get(std.NUM_REQUESTS)
        if num_requests is not None:
            return f"{num_nodes} nodes, {num_requests} requests"
        else:
            return f"{num_nodes} nodes"

    async for msg in stream:
        match msg:
            case dp.Yield():
                success = True
            case dp.Spent(b):
                total_budget += b
            case dp.Barrier():
                pass
        await task.set_result(compute_result())
        await task.set_status(compute_status())
    await task.set_result(compute_result())
