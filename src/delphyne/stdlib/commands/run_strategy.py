"""
Standard commands for running strategies.
"""

import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, assert_type, cast

import delphyne.analysis as analysis
import delphyne.analysis.feedback as fb
import delphyne.core_and_base as dp
import delphyne.stdlib.environments as en
import delphyne.stdlib.models as md
import delphyne.stdlib.policies as pol
import delphyne.stdlib.tasks as ta
import delphyne.utils.caching as ca
from delphyne.core import irefs, refs
from delphyne.core.streams import Barrier, Solution, Spent
from delphyne.stdlib.globals import stdlib_globals
from delphyne.utils.typing import pydantic_dump


@dataclass(kw_only=True)
class RunStrategyResponse:
    """
    Response type for the `run_strategy` command.

    Attributes:
        success: Whether at least one success value was generated.
        values: Generated success values.
        spent_budget: Spent budget.
        success_nodes: Identifiers of success nodes in the trace. Only
            available when the trace is exported, in which case it has
            the same length as `values`.
        raw_trace: Raw trace of the strategy execution, if requested.
        log: Log messages generated during the strategy execution.
        browsable_trace: A browsable trace, if requested.
    """

    success: bool
    values: Sequence[Any | None]
    spent_budget: Mapping[str, float]
    success_nodes: Sequence[int] | None = None
    raw_trace: dp.ExportableTrace | None = None
    log: Sequence[dp.ExportableLogMessage] | None = None
    browsable_trace: fb.Trace | None = None


@dataclass(kw_only=True)
class RunLoadedStrategyArgs[N: dp.Node, P, T]:
    """
    Arguments for the `run_loaded_strategy` command.

    See `RunStrategyArgs` for details.
    """

    strategy: dp.StrategyComp[N, P, T]
    policy: pol.StandardPolicy[N, P]
    answer_loader: dp.AnswerLoader | None
    num_generated: int = 1
    budget: dict[str, float] | None = None
    using: Sequence[dp.AnswerSource] | None = None
    cache_file: str | None = None
    embeddings_cache_file: str | None = None
    cache_mode: ca.CacheMode = "read_write"
    log_level: dp.LogLevel = "info"
    log_long_computations: tuple[dp.LogLevel, float] | None = None
    export_raw_trace: bool = True
    export_log: bool = True
    export_browsable_trace: bool = True
    export_all_on_pull: bool = False
    remove_timing_info: bool = False


def run_loaded_strategy_with_caches[N: dp.Node, P, T](
    task: ta.TaskContext[ta.CommandResult[RunStrategyResponse]],
    exe: ta.ExecutionContext,
    args: RunLoadedStrategyArgs[N, P, T],
    request_cache: md.LLMCache | None,
    embeddings_cache: dp.EmbeddingsCache | None,
):
    answer_database = None
    if args.using is not None:
        assert exe.workspace_root is not None, (
            "No workspace root is specified."
        )
        assert args.answer_loader is not None, "No answer loader specified."
        answer_database = dp.AnswerDatabase(
            args.using, loader=args.answer_loader
        )
    env = en.PolicyEnv(
        prompt_dirs=exe.prompt_dirs,
        data_dirs=exe.data_dirs,
        demonstration_files=exe.demo_files,
        cache=request_cache,
        embeddings_cache=embeddings_cache,
        global_embeddings_cache_file=exe.global_embeddings_cache_file,
        object_loader=exe.object_loader(extra_objects=stdlib_globals()),
        override_answers=answer_database,
        log_level=args.log_level,
        log_long_computations=args.log_long_computations,
    )
    lock = threading.Lock()  # to protect state that can be pulled
    # We do not need to cache all tree nodes (which can cost a lot of
    # memory) unless a browsable trace may be computed.
    cache: dp.TreeCache | None = None
    hooks: list[dp.TreeHook] = []
    if args.export_browsable_trace or args.export_all_on_pull:
        cache = {}
    if (
        args.export_browsable_trace
        or args.export_all_on_pull
        or args.export_raw_trace
    ):
        hooks.append(dp.tracer_hook(env.tracer))
    monitor = dp.TreeMonitor(cache, hooks=hooks)
    tree = dp.reify(args.strategy, monitor)
    policy = args.policy
    stream = policy(tree, env)
    if args.budget is not None:
        stream = stream.with_budget(dp.BudgetLimit(args.budget))
    stream = stream.take(args.num_generated)
    results: list[dp.Tracked[T]] = []
    success = False
    total_budget = dp.Budget.zero()

    def serialize_result(res: T) -> Any | None:
        ret_type = args.strategy.return_type()
        return pydantic_dump(ret_type, res)

    def compute_result(
        export_all: bool = False,
    ) -> ta.CommandResult[RunStrategyResponse]:
        export_raw_trace = args.export_raw_trace or export_all
        export_browsable_trace = args.export_browsable_trace or export_all
        export_log = args.export_log or export_all

        trace = env.tracer.trace
        raw_trace = trace.export() if export_raw_trace else None
        browsable_trace: fb.Trace | None = None
        success_nodes = None
        if raw_trace is not None:
            success_nodes = [
                _node_id_of_tracked_value(r, trace).id for r in results
            ]
        if export_browsable_trace:
            assert cache is not None
            browsable_trace = analysis.compute_browsable_trace(
                trace, cache=cache
            )
        log = None
        if export_log:
            log = list(
                env.tracer.export_log(
                    remove_timing_info=args.remove_timing_info
                )
            )
        values = [serialize_result(r.value) for r in results]
        response = RunStrategyResponse(
            success=success,
            values=values,
            spent_budget=total_budget.values,
            raw_trace=raw_trace,
            success_nodes=success_nodes,
            log=log,
            browsable_trace=browsable_trace,
        )
        return ta.CommandResult([], response)

    def compute_status():
        num_nodes = len(env.tracer.trace.nodes)
        num_requests = total_budget.values.get(md.NUM_REQUESTS)
        price = total_budget.values.get(md.DOLLAR_PRICE)

        ret: list[str] = [f"{num_nodes} nodes"]
        if num_requests is not None:
            # If num_requests is a float equal to an int, cast to int
            # for display
            if isinstance(num_requests, float) and num_requests.is_integer():
                num_requests = int(num_requests)
            ret += [f"{num_requests} requests"]
        if price is not None:
            price *= 100  # in cents
            ret += [f"{price:.2g}¢"]

        return ", ".join(ret)

    def pull_status() -> str:
        with lock:
            return compute_status()

    def pull_result() -> ta.CommandResult[RunStrategyResponse]:
        with lock:
            return compute_result(args.export_all_on_pull)

    task.set_pull_status(pull_status)
    task.set_pull_result(pull_result)

    last_refreshed_result = time.time()
    last_refreshed_status = time.time()
    # TODO: generating each element is blocking here. Should we spawn a
    # thread for every new element?
    for msg in stream:
        with lock:
            match msg:
                case Solution():
                    success = True
                    results.append(msg.tracked)
                case Spent(b):
                    total_budget += b
                case Barrier():
                    pass
        interrupted = task.interruption_requested()
        if interrupted or (
            exe.result_refresh_period is not None
            and time.time() - last_refreshed_result > exe.result_refresh_period
        ):
            task.set_result(compute_result())
            last_refreshed_result = time.time()
        if (
            exe.status_refresh_period is not None
            and time.time() - last_refreshed_status > exe.status_refresh_period
        ):
            task.set_status(compute_status())
            last_refreshed_status = time.time()
        if interrupted:
            break
    task.set_result(compute_result())


def run_loaded_strategy[N: dp.Node, P, T](
    task: ta.TaskContext[ta.CommandResult[RunStrategyResponse]],
    exe: ta.ExecutionContext,
    args: RunLoadedStrategyArgs[N, P, T],
):
    """
    Command for running an oracular program.
    """

    with_cache_spec(
        partial(run_loaded_strategy_with_caches, task, exe, args),
        cache_root=exe.cache_root,
        cache_file=args.cache_file,
        embeddings_cache_file=args.embeddings_cache_file,
        cache_mode=args.cache_mode,
    )


def with_cache_spec[T](
    f: Callable[[dp.LLMCache | None, dp.EmbeddingsCache | None], T],
    *,
    cache_root: Path | None,
    cache_file: str | None,
    embeddings_cache_file: str | None,
    cache_mode: ca.CacheMode,
) -> T:
    """
    Utility function for loading request and embeddings caches
    before calling `f`.
    """

    request_cache_path = None
    embeddings_cache_path = None
    unspecified_message = "Nonspecified cache root."

    if cache_file is not None:
        assert cache_root is not None, unspecified_message
        request_cache_path = cache_root / cache_file
    if embeddings_cache_file is not None:
        assert cache_root is not None, unspecified_message
        embeddings_cache_path = cache_root / embeddings_cache_file
    with md.load_optional_request_cache(
        request_cache_path, mode=cache_mode
    ) as request_cache:
        with dp.load_optional_embeddings_cache(
            embeddings_cache_path, mode=cache_mode
        ) as embeddings_cache:
            return f(request_cache, embeddings_cache)


@dataclass(kw_only=True)
class RunStrategyArgs:
    """
    Arguments for the `run_strategy` command that runs an oracular
    program.

    Attributes:
        strategy: Name of the strategy to run.
        args: Arguments to pass to the strategy constructor.
        policy: Name of the policy to use.
        policy_args: Arguments to pass to the policy constructor.
        num_generated: Number of success values to generate.
        budget: Budget limit (infinite for unspecified metrics).
        using: Answer sources to use for overriding LLM oracles (see
            `PolicyEnv`). This is particularly useful for
            auto-completing demonstrations.
        cache_file: File within the global cache directory to use for
            request caching, or `None` to disable caching.
        embeddings_cache_file: File within the global cache directory to
            use for embeddings caching, or `None` to disable embeddings
            caching.
        cache_mode: Cache mode to use.
        log_level: Minimum log level to record. Messages with a lower
            level will be ignored.
        log_long_computations: If set, log every computation taking
            more than the given number of seconds at the given severity
            level.
        export_raw_trace: Whether to export the raw execution trace.
        export_log: Whether to export the log messages.
        export_browsable_trace: Whether to export a browsable trace,
            which can be visualized in the VSCode extension (see
            [delphyne.analysis.feedback.Trace][]).
        export_all_on_pull: Whether to export all
            information (raw trace, log, browsable trace) when an
            intermediate result is pulled.
        remove_timing_info: Remove all timing information from the
            result file (e.g. timestamp for each log message). This is
            particularly useful for making commands deterministic.
    """

    strategy: str
    args: dict[str, object]
    policy: str
    policy_args: dict[str, object]
    budget: dict[str, float]
    using: Sequence[dp.AnswerSource] | None = None
    num_generated: int = 1
    cache_file: str | None = None
    embeddings_cache_file: str | None = None
    cache_mode: ca.CacheMode = "read_write"
    log_level: dp.LogLevel = "info"
    log_long_computations: tuple[dp.LogLevel, float] | None = None
    export_raw_trace: bool = True
    export_log: bool = True
    export_browsable_trace: bool = True
    export_all_on_pull: bool = False
    remove_timing_info: bool = False


def run_strategy(
    task: ta.TaskContext[ta.CommandResult[RunStrategyResponse]],
    exe: ta.ExecutionContext,
    args: RunStrategyArgs,
):
    """
    Command for running an oracular program from a serialized
    specification.
    """
    loader = exe.object_loader(extra_objects=stdlib_globals())
    strategy = loader.load_strategy_instance(args.strategy, args.args)
    policy = loader.load_and_call_function(args.policy, args.policy_args)
    answer_loader = None
    if args.using is not None:
        assert exe.workspace_root is not None
        answer_loader = dp.standard_answer_loader(exe.workspace_root, loader)
    assert isinstance(policy, dp.AbstractPolicy)
    policy = cast(pol.StandardPolicy[Any, Any], policy)
    run_loaded_strategy(
        task=task,
        exe=exe,
        args=RunLoadedStrategyArgs(
            strategy=strategy,
            policy=policy,
            answer_loader=answer_loader,
            num_generated=args.num_generated,
            budget=args.budget,
            using=args.using,
            cache_file=args.cache_file,
            embeddings_cache_file=args.embeddings_cache_file,
            cache_mode=args.cache_mode,
            log_level=args.log_level,
            log_long_computations=args.log_long_computations,
            export_raw_trace=args.export_raw_trace,
            export_log=args.export_log,
            export_browsable_trace=args.export_browsable_trace,
            export_all_on_pull=args.export_all_on_pull,
            remove_timing_info=args.remove_timing_info,
        ),
    )


def _node_id_of_tracked_value(
    value: dp.Tracked[object], trace: dp.Trace
) -> irefs.NodeId:
    ref = value.ref
    while isinstance(ref, refs.IndexedRef):
        ref = ref.ref
    eref = ref.element
    assert not isinstance(eref, refs.Answer)
    assert_type(eref, refs.NodePath)
    main_space = refs.GlobalSpacePath(())
    gref = refs.GlobalNodeRef(main_space, eref)
    return trace.convert_global_node_ref(gref)
