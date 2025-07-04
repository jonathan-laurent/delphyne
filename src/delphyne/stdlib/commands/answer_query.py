"""
Standard commands for answering queries.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import delphyne.analysis as analysis
import delphyne.core as dp
import delphyne.stdlib as std
import delphyne.stdlib.models as mo
import delphyne.stdlib.tasks as ta

DEFAULT_OPENAI_MODEL = "gpt-4o"


@dataclass
class AnswerQueryArgs:
    query: str
    args: dict[str, object]
    prompt_only: bool
    model: str | None = None
    num_answers: int = 1
    iterative_mode: bool = False
    budget: dict[str, float] | None = None


@dataclass
class AnswerQueryResponse:
    num_successes: int
    log: Sequence[dp.ExportableLogMessage]


async def answer_query(
    task: ta.TaskContext[ta.CommandResult[AnswerQueryResponse]],
    exe: ta.CommandExecutionContext,
    cmd: AnswerQueryArgs,
):
    # TODO: no examples for now. Also, we have to externalize this anyway.
    loader = analysis.ObjectLoader(exe.base)
    query = loader.load_query(cmd.query, cmd.args)
    env = dp.PolicyEnv(
        prompt_dirs=exe.prompt_dirs,
        data_dirs=exe.data_dirs,
        demonstration_files=exe.demo_files,
        do_not_match_identical_queries=True,
    )
    attached = dp.spawn_standalone_query(query)
    model = std.openai_model(cmd.model or DEFAULT_OPENAI_MODEL)
    if cmd.prompt_only:
        model = mo.DummyModel()
    policy = std.few_shot(
        model=model, enable_logging=True, iterative_mode=cmd.iterative_mode
    )
    stream = policy(attached, env)
    if cmd.budget is not None:
        stream = std.stream_with_budget(stream, dp.BudgetLimit(cmd.budget))
    if cmd.prompt_only:
        only_one_req = dp.BudgetLimit({std.NUM_REQUESTS: 1})
        stream = std.stream_with_budget(stream, only_one_req)
    stream = std.stream_take(stream, cmd.num_answers)
    total_budget = dp.Budget.zero()
    num_successes = 0

    def compute_status():
        num_requests = total_budget.values.get(std.NUM_REQUESTS)
        if num_requests is not None:
            return f"{num_successes} success(es), {num_requests} request(s)"
        return ""

    def compute_result():
        log = list(env.tracer.export_log())
        resp = AnswerQueryResponse(num_successes, log)
        return ta.CommandResult([], resp)

    for msg in stream:
        match msg:
            case dp.Yield():
                num_successes += 1
            case dp.Spent(b):
                total_budget += b
            case dp.Barrier():
                pass
        await task.set_result(compute_result())
        await task.set_status(compute_status())
    await task.set_result(compute_result())
