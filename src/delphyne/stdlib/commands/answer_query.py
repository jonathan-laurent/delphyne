"""
Standard commands for answering queries.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial

import delphyne.core_and_base as dp
import delphyne.stdlib.environments as en
import delphyne.stdlib.models as md
import delphyne.stdlib.models as mo
import delphyne.stdlib.queries as qu
import delphyne.stdlib.standard_models as stdm
import delphyne.stdlib.tasks as ta
import delphyne.utils.caching as ca
from delphyne.core.streams import Barrier, Solution, Spent
from delphyne.stdlib.commands.run_strategy import with_cache_spec
from delphyne.stdlib.globals import stdlib_globals

DEFAULT_MODEL_NAME = "gpt-4o"


@dataclass(kw_only=True)
class AnswerQueryArgs:
    """
    Arguments for the `answer_query` command.

    Attributes:
        query: The name of the query to answer.
        args: Arguments for the query, as a dictionary of JSON values.
        prompt_only: If `True`, a dummy model is used that always
            errors, so that only the prompt can be seen in the logs.
        model: The name of the model to use for answering the query.
        num_answers: The number of answers to generate.
        iterative_mode: Whether to answer the query in iterative mode
            (see `few_shot` for details).
        budget: Budget limit (infinite for unspecified metrics).
        cache_file: File within the global cache directory to use for
            request caching, or `None` to disable caching.
        cache_mode: Cache mode to use.
        log_level: Minimum log level to record. Messages with a lower
            level will be ignored.
    """

    query: str
    args: dict[str, object]
    prompt_only: bool
    model: str | None = None
    num_answers: int = 1
    iterative_mode: bool = False
    budget: dict[str, float] | None = None
    cache_file: str | None = None
    embeddings_cache_file: str | None = None
    cache_mode: ca.CacheMode = "read_write"
    log_level: dp.LogLevel = "info"


@dataclass
class AnswerQueryResponse:
    num_successes: int
    log: Sequence[dp.ExportableLogMessage]


def answer_query_with_cache(
    task: ta.TaskContext[ta.CommandResult[AnswerQueryResponse]],
    exe: ta.CommandExecutionContext,
    cmd: AnswerQueryArgs,
    cache: md.LLMCache | None,
    embeddings_cache: dp.EmbeddingsCache | None,
):
    loader = exe.object_loader(extra_objects=stdlib_globals())
    query = loader.load_query(cmd.query, cmd.args)
    env = en.PolicyEnv(
        object_loader=loader,
        prompt_dirs=exe.prompt_dirs,
        data_dirs=exe.data_dirs,
        demonstration_files=exe.demo_files,
        cache=cache,
        embeddings_cache=embeddings_cache,
        log_level=cmd.log_level,
    )
    attached = dp.spawn_standalone_query(query)
    model_name = cmd.model or DEFAULT_MODEL_NAME
    model = stdm.standard_model(model_name)
    if cmd.prompt_only:
        model = mo.DummyModel()
    policy = qu.few_shot(
        model=model,
        iterative_mode=cmd.iterative_mode,
        # We exclude examples related to the exact same query, since
        # `answer_query` is typically used to inepect model behavior and
        # there is little to see if the answer is directly copied from
        # an example.
        select_examples=dp.all_examples.exclude_identical_queries(),
    )
    stream = policy(attached, env)
    if cmd.budget is not None:
        stream = stream.with_budget(dp.BudgetLimit(cmd.budget))
    if cmd.prompt_only:
        only_one_req = dp.BudgetLimit({mo.NUM_REQUESTS: 1})
        stream = stream.with_budget(only_one_req)
    stream = stream.take(cmd.num_answers)
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
    """
    A command for answering a query. See `AnswerQueryArgs`.
    """
    with_cache_spec(
        partial(answer_query_with_cache, task, exe, cmd),
        cache_root=exe.cache_root,
        cache_file=cmd.cache_file,
        embeddings_cache_file=cmd.embeddings_cache_file,
        cache_mode=cmd.cache_mode,
    )
