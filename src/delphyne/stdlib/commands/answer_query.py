"""
Standard commands for answering queries.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import delphyne.analysis as analysis
import delphyne.core as dp
import delphyne.stdlib.models as md
import delphyne.stdlib.models as mo
import delphyne.stdlib.queries as qu
import delphyne.stdlib.standard_models as stdm
import delphyne.stdlib.streams as st
import delphyne.stdlib.tasks as ta
import delphyne.utils.caching as ca
from delphyne.core.streams import Barrier, Solution, Spent
from delphyne.stdlib.commands.run_strategy import CacheFormat, with_cache_spec

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
    cache_dir: str | None = None
    cache_mode: ca.CacheMode = "read_write"
    cache_format: CacheFormat = "yaml"


@dataclass
class AnswerQueryResponse:
    num_successes: int
    log: Sequence[dp.ExportableLogMessage]


def answer_query_with_cache(
    task: ta.TaskContext[ta.CommandResult[AnswerQueryResponse]],
    exe: ta.CommandExecutionContext,
    cmd: AnswerQueryArgs,
    cache_spec: ca.CacheSpec | None,
):
    # TODO: no examples for now. Also, we have to externalize this anyway.
    loader = analysis.ObjectLoader(exe.base)
    query = loader.load_query(cmd.query, cmd.args)
    env = dp.PolicyEnv(
        prompt_dirs=exe.prompt_dirs,
        data_dirs=exe.data_dirs,
        demonstration_files=exe.demo_files,
        do_not_match_identical_queries=True,
        cache=cache_spec,
        make_cache=md.LLMCache,
    )
    attached = dp.spawn_standalone_query(query)
    model = stdm.openai_model(cmd.model or DEFAULT_OPENAI_MODEL)
    if cmd.prompt_only:
        model = mo.DummyModel()
    policy = qu.few_shot(
        model=model, enable_logging=True, iterative_mode=cmd.iterative_mode
    )
    stream = policy(attached, env).gen()
    if cmd.budget is not None:
        stream = st.stream_with_budget(stream, dp.BudgetLimit(cmd.budget))
    if cmd.prompt_only:
        only_one_req = dp.BudgetLimit({mo.NUM_REQUESTS: 1})
        stream = st.stream_with_budget(stream, only_one_req)
    stream = st.stream_take(stream, cmd.num_answers)
    total_budget = dp.Budget.zero()
    num_successes = 0

    def compute_status():
        num_requests = total_budget.values.get(mo.NUM_REQUESTS)
        if num_requests is not None:
            return f"{num_successes} success(es), {num_requests} request(s)"
        return ""

    def compute_result():
        log = list(env.tracer.export_log())
        resp = AnswerQueryResponse(num_successes, log)
        return ta.CommandResult([], resp)

    for msg in stream:
        match msg:
            case Solution():
                num_successes += 1
            case Spent(b):
                total_budget += b
            case Barrier():
                pass
        task.set_result(compute_result())
        task.set_status(compute_status())
    task.set_result(compute_result())


def answer_query(
    task: ta.TaskContext[ta.CommandResult[AnswerQueryResponse]],
    exe: ta.CommandExecutionContext,
    cmd: AnswerQueryArgs,
):
    with_cache_spec(
        partial(answer_query_with_cache, task, exe, cmd),
        cache_root=exe.cache_root,
        cache_dir=cmd.cache_dir,
        cache_mode=cmd.cache_mode,
        cache_format=cmd.cache_format,
    )
