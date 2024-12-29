"""
Custom Delphyne Commands
"""

import asyncio
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from delphyne.core.inspect import underlying_strategy_param_type
from delphyne.core.queries import Prompt, PromptOptions
from delphyne.core.refs import AnswerId
from delphyne.core.strategies import StrategyTree
from delphyne.core.tracing import ExportableTrace
from delphyne.server.evaluate_demo import (
    ExecutionContext,
    compute_browsable_trace,
)
from delphyne.server.feedback import Diagnostic, Trace
from delphyne.server.navigation import NavigationTree
from delphyne.server.tasks import TaskContext
from delphyne.stdlib.dsl import WrappedStrategy
from delphyne.stdlib.generators import Budget, BudgetCounter, GenEnv
from delphyne.stdlib.openai_util import stream_openai_response
from delphyne.stdlib.search_envs import HasSearchEnv, Params
from delphyne.utils.typing import TypeAnnot, pydantic_load


@dataclass
class CommandExecutionContext:
    base: ExecutionContext
    demo_files: list[Path]


@dataclass
class CommandSpec:
    command: str
    args: dict[str, object]


@dataclass
class CommandResult[T]:
    diagnostics: list[Diagnostic]
    result: T


async def execute_command(
    task: TaskContext[CommandResult[Any]],
    exe: CommandExecutionContext,
    cmd: CommandSpec,
):
    try:
        match cmd.command:
            case "test_command":
                await test_command(task)
            case "answer_query":
                payload = pydantic_load(AnswerQueryCmd, cmd.args)
                await answer_query(task, exe, payload)
            case "run_strategy":
                payload = pydantic_load(RunStrategyCmd, cmd.args)
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


async def test_command(task: TaskContext[CommandResult[list[int]]]):
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
    options: PromptOptions
    args: dict[str, object]


@dataclass
class AnswerQueryResponse:
    prompt: Prompt | None
    response: str | None


def load_params[P: HasSearchEnv](
    exe: CommandExecutionContext,
    type: TypeAnnot[P],
    serialized: dict[str, object],
) -> P:
    serialized |= {
        "env": {
            "demo_paths": exe.demo_files,
            "exe_context": {
                "strategy_dirs": exe.base.strategy_dirs,
                "modules": exe.base.modules,
            },
        }
    }
    return pydantic_load(type, serialized)


async def answer_query(
    task: TaskContext[CommandResult[AnswerQueryResponse]],
    exe: CommandExecutionContext,
    cmd: AnswerQueryCmd,
):
    query = exe.base.load_query(cmd.query, cmd.args)
    param_type = cast(Any, query.param_type())
    params = load_params(exe, param_type, cmd.params)
    examples = params.env.collect_examples(query)
    prompt = query.create_prompt(params, examples)
    prompt.options.__dict__.update(cmd.options.__dict__)
    await task.set_result(CommandResult([], AnswerQueryResponse(prompt, None)))
    if cmd.prompt_only:
        return
    answer = ""
    async for chunk in stream_openai_response(prompt):
        answer += chunk
        await task.set_result(
            CommandResult([], AnswerQueryResponse(prompt, answer))
        )


#####
##### Run Search
#####


@dataclass
class BudgetLimit:
    num_requests: float | None = None
    num_context_tokens: float | None = None
    num_generated_tokens: float | None = None
    price: float | None = None


@dataclass
class RunStrategyCmd:
    strategy: str
    args: dict[str, object]
    params: dict[str, object]
    budget: BudgetLimit


@dataclass
class RunStrategyResponse:
    success: bool
    raw_trace: ExportableTrace
    prompts: dict[int, Prompt]
    browsable_trace: Trace


async def run_strategy(
    task: TaskContext[CommandResult[RunStrategyResponse]],
    exe: CommandExecutionContext,
    cmd: RunStrategyCmd,
):
    strategy = exe.base.find_and_instantiate_wrapped_strategy(
        cmd.strategy, cmd.args
    )
    param_type = underlying_strategy_param_type(strategy)
    params = cast(Params, load_params(exe, param_type, cmd.params))
    tree = NavigationTree.make(StrategyTree.new(strategy))
    strategy = cast(WrappedStrategy[Any, Any, Any], strategy)
    search = strategy.search_policy
    assert search is not None
    counter = BudgetCounter(Budget.limit(**cmd.budget.__dict__))
    gen_env = GenEnv([counter], "lazy")
    gen = search(gen_env, tree, params)

    prompts: dict[int, Prompt] = {}

    def register_prompt(p: Prompt, aid: AnswerId):
        prompts[aid.id] = p

    params.env.set_prompt_hook(register_prompt)

    def compute_result():
        raw_trace = tree.tracer.export()
        browsable_trace = compute_browsable_trace(tree, None)
        response = RunStrategyResponse(
            success, raw_trace, prompts, browsable_trace
        )
        return CommandResult([], response)

    def compute_status():
        num_nodes = len(tree.tracer.nodes)
        num_requests = counter.spent.num_requests
        return f"{num_nodes} nodes, {num_requests} requests"

    success = False
    await task.set_status(compute_status())
    async for resp in gen:
        if resp.items:
            success = True
            break
        if not gen_env.can_spend(Budget.spent(num_requests=1)):
            break
        await task.set_result(compute_result())
        await task.set_status(compute_status())
    await task.set_result(compute_result())
