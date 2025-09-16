"""
Delphyne Tasks.

In order to better integrate with the UI, we propose a standard
interface for defining tasks that can output a stream of status messages
along with intermediate results.
"""

import inspect
import os
import time
import traceback
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Concatenate, Literal, Never, Protocol

import delphyne.analysis as analysis
import delphyne.analysis.feedback as fb
import delphyne.stdlib.tasks as ta
import delphyne.utils.typing as ty
from delphyne.utils.yaml import pretty_yaml

DEFAULT_STRATEGY_DIRS = (Path("."),)
DEFAULT_PROMPTS_DIRS = (Path("prompts"),)
DEFAULT_DATA_DIRS = (Path("data"),)


#####
##### Tasks
#####


class TaskContext[T](typing.Protocol):
    """
    Context object accessible to all tasks for reporting progress and
    intermediate results.

    Most tasks operate by _pushing_ status messages and intermediate (or
    final) results using `set_status` and `set_result` respectively.
    Optionally, tasks can also register functions that can be used for
    _pulling_ status messages and results from the outside
    (`set_pull_status` and `set_pull_result`).
    """

    # To be called externally
    def interrupt(self) -> None: ...

    # To be called internally
    def interruption_requested(self) -> bool: ...
    def log(self, message: str) -> None: ...
    def set_status(self, message: str) -> None: ...
    def set_result(self, result: T) -> None: ...
    def raise_internal_error(self, message: str) -> None: ...
    def set_pull_status(self, pull: "PullStatusFn") -> None: ...
    def set_pull_result(self, pull: "PullResultFn[T]") -> None: ...


type PullResultFn[T] = Callable[[], T | None]


type PullStatusFn = Callable[[], str | None]


type StreamingTask[**P, T] = Callable[Concatenate[TaskContext[T], P], None]


type TaskMessage[T] = (
    tuple[Literal["log"], str]
    | tuple[Literal["set_status"], str]
    | tuple[Literal["set_result"], T]
    | tuple[Literal["internal_error"], str]
)


def stream_of_fun[**P, T](f: Callable[P, T]) -> StreamingTask[P, T]:
    """
    Convert a function into a degenerate streaming task.
    """

    def stream(context: TaskContext[T], *args: P.args, **kwargs: P.kwargs):
        try:
            ret = f(*args, **kwargs)
            context.set_result(ret)
        except Exception:
            context.raise_internal_error(traceback.format_exc())

    return stream


def counting_generator(task: TaskContext[Never], n: int):
    """
    A basic task example, to be used for testing.
    """
    try:
        for i in range(n):
            time.sleep(1)
            print(f"Send: {i}", flush=True)
            task.set_status(str(i))
            task.log(f"Sent: {i}")
            i += 1
    finally:
        print("Generator over.", flush=True)


#####
##### Particular Case: Commands
#####


@dataclass(frozen=True, kw_only=True)
class CommandExecutionContext:
    """
    Context information available to all commands.

    This information is usually specified in `delphyne.yaml` project
    files, and also locally in `@config` blocks. All values have
    defaults. All paths are usually expressed relative to a single
    workspace root directory, and can be made absolute using the
    `with_root` method.

    Parameters:
        strategy_dirs: A sequence of directories in which strategy
            modules can be found.
        modules: A sequence of module in which Python object identifiers
            can be resolved (module names can feature `.`).
        demo_files: A sequence of demonstration files (either including or
            excluding the `*.demo.yaml` extension).
        prompt_dirs: A sequence of directories in which to look for
            prompt templates.
        data_dirs: A sequence of directories in which to look for data
            files.
        cache_root: The directory in which to store all request cache
            subdirectories.
        result_refresh_period: The period in seconds at which the
            current result is computed and communicated to the UI (e.g.,
            the period at which the current trace is exported when
            running oracular programs). If `None`, the result is never
            refreshed (until the command terminates).
        status_refresh_period: The period in seconds at which the
            current status message is communicated to the UI. If `None`,
            the status is never refreshed (until the command
            terminates).
        workspace_root: The root directory of the workspace. This value
            is not meant to be provided in configuration files. Rather,
            it is set when using the `with_root` method. This setting is
            useful for interpreting some relative paths in demonstration
            and command files (e.g. `using` directives).

    !!! info "Local conguration blocks"
        Demonstration and command files can override some configuration
        information from the `delphyne.yaml` file by featuring a comment
        block such as:

        ```yaml
        # @config
        # modules: ["my_strategy_module"]
        # demo_files: ["demo.yaml"]
        # @end
        ```

        The comment block must be placed at the start of the file,
        possibly after other comments.
    """

    strategy_dirs: Sequence[Path] = DEFAULT_STRATEGY_DIRS
    modules: Sequence[str] = ()
    demo_files: Sequence[Path] = ()
    prompt_dirs: Sequence[Path] = DEFAULT_PROMPTS_DIRS
    data_dirs: Sequence[Path] = DEFAULT_DATA_DIRS
    cache_root: Path | None = None
    result_refresh_period: float | None = None
    status_refresh_period: float | None = None
    workspace_root: Path | None = None

    @property
    def base(self) -> analysis.DemoExecutionContext:
        """
        Obtain a demonstration execution context.
        """
        return analysis.DemoExecutionContext(
            strategy_dirs=self.strategy_dirs,
            modules=self.modules,
            workspace_root=self.workspace_root,
        )

    def with_root(self, root: Path) -> "CommandExecutionContext":
        """
        Make all paths absolute given a path to the workspace root.
        """
        return CommandExecutionContext(
            modules=self.modules,
            demo_files=[root / f for f in self.demo_files],
            strategy_dirs=[root / d for d in self.strategy_dirs],
            prompt_dirs=[root / d for d in self.prompt_dirs],
            data_dirs=[root / d for d in self.data_dirs],
            cache_root=None if self.cache_root is None else self.cache_root,
            result_refresh_period=self.result_refresh_period,
            status_refresh_period=self.status_refresh_period,
            workspace_root=root,
        )


# Somehow, pyright sometimes infers the wrong variance for `T` in
# `CommandResult` so we specify it manually.
T = typing.TypeVar("T", covariant=True)


@dataclass(frozen=True)
class CommandResult(typing.Generic[T]):
    """
    Outcome of executing a command.
    """

    diagnostics: Sequence[fb.Diagnostic]
    result: T


class Command[A, T](Protocol):
    def __call__(
        self,
        task: ta.TaskContext[CommandResult[T]],
        exe: CommandExecutionContext,
        args: A,
    ) -> None:
        """
        A command is a special task which, in addition to emitting
        status messages and intermediate results, has access to a global
        configuration context and produces a set of diagnostics.

        Special support is provided for running commands in the Delphyne
        CLI and in the VSCode extension.
        """
        ...


def command_args_type(cmd: Command[Any, Any]) -> Any:
    sig = inspect.signature(cmd)
    parameters = list(sig.parameters.keys())
    assert len(parameters) == 3, f"Invalid command: {cmd}"
    hints = typing.get_type_hints(cmd)
    return hints[parameters[-1]]


def command_result_type(cmd: Command[Any, Any]) -> Any:
    """
    Inspect a command's return type `T` by matching the type annotation
    of its first argument against `TaskContext[CommandResult[T]]`.
    """
    sig = inspect.signature(cmd)
    parameters = list(sig.parameters.keys())
    assert len(parameters) == 3
    hints = typing.get_type_hints(cmd)
    task_context_type: Any = hints[parameters[0]]
    assert typing.get_origin(task_context_type) == TaskContext
    cmd_res_type = typing.get_args(task_context_type)[0]
    assert typing.get_origin(cmd_res_type) == CommandResult
    return typing.get_args(cmd_res_type)[0]


def command_optional_result_wrapper_type(
    cmd: Command[Any, Any],
) -> type[CommandResult[Any | None]]:
    """
    Return `CommandResult[T | None]` where `T = command_result_type(cmd)`.
    """
    ret_type = command_result_type(cmd)
    return CommandResult[ret_type | None]


def command_name(cmd: Command[Any, Any]) -> str:
    return cmd.__name__  # type: ignore


def run_command[A, T](
    command: Command[A, T],
    args: A,
    ctx: CommandExecutionContext,
    dump_statuses: Path | None = None,
    dump_result: Path | None = None,
    dump_log: Path | None = None,
    on_status: Callable[[str], None] | None = None,
    on_set_pull_status: Callable[[PullStatusFn], None] | None = None,
    on_set_pull_result: Callable[[PullResultFn[CommandResult[T]]], None]
    | None = None,
    on_set_pull_result_str: Callable[[Callable[[], str]], None] | None = None,
    add_header: bool = True,
    handle_sigint: bool = False,
) -> CommandResult[T | None]:
    """
    A simple command runner.

    Arguments:
        command: The command to run.
        args: The arguments to pass to the command.
        ctx: The command execution context.
        dump_statuses: A file in which to dump status messages at
            regular intervals.
        dump_result: A file in which to dump intermediate and final
            results, whose content is refreshed at regular intervals.
        dump_log: A file in which to dump log messages.
        on_status: A function to call every time a status message is
            issued.
        on_set_pull_status: A function to call if and when the task
            registers a pull function for status messages.
        on_set_pull_result: A function to call if and when the task
            registers a pull function for results.
        on_set_pull_result_str: Similar to `on_set_pull_result`, but
            produces strings instead of `CommandResult` objects.
        add_header: If `True`, the dumped result is prefixed with a
            header containing the command name and arguments.
        handle_sigint: If `True`, pressing `Ctrl+C` sends the command
            task an interruption request by calling
            `TaskContext.interrupt`, allowing it to gracefully terminate
            instead of abruptly interrupting it.

    Non-existing directories are created automatically.
    """

    def _result_to_string(result: CommandResult[T | None]) -> str:
        ret_ty = command_optional_result_wrapper_type(command)
        ret: Any = ty.pydantic_dump(ret_ty, result)
        if add_header:
            args_type = command_args_type(command)
            ret = {
                "command": command_name(command),
                "args": ty.pydantic_dump(args_type, args),
                "outcome": ret,
            }
        return "# delphyne-command\n\n" + pretty_yaml(ret)

    class Handler:
        def __init__(self):
            self.result: CommandResult[T | None] = CommandResult([], None)
            self.interrupted = False

        def interrupt(self) -> None:
            self.interrupted = True

        def interruption_requested(self) -> bool:
            return self.interrupted

        def log(self, message: str) -> None:
            if dump_log is not None:
                os.makedirs(dump_log.parent, exist_ok=True)
                with open(dump_log, "a") as f:
                    f.write(message + "\n")

        def set_status(self, message: str) -> None:
            if dump_statuses is not None:
                os.makedirs(dump_statuses.parent, exist_ok=True)
                with open(dump_statuses, "a") as f:
                    f.write(message + "\n")
            if on_status is not None:
                on_status(message)

        def set_result(self, result: CommandResult[T]) -> None:
            self.result = result
            if dump_result is not None:
                ret = _result_to_string(result)
                os.makedirs(dump_result.parent, exist_ok=True)
                with open(dump_result, "w") as f:
                    f.write(ret)

        def set_pull_status(self, pull: PullStatusFn) -> None:
            if on_set_pull_status is not None:
                on_set_pull_status(pull)

        def set_pull_result(
            self, pull: PullResultFn[CommandResult[T]]
        ) -> None:
            if on_set_pull_result is not None:
                on_set_pull_result(pull)
            if on_set_pull_result_str is not None:

                def pull_str() -> str:
                    ret = pull()
                    if ret is None:
                        ret = CommandResult([], None)
                    return _result_to_string(ret)

                on_set_pull_result_str(pull_str)

        def raise_internal_error(self, message: str) -> None:
            error = ("error", f"Internal error: {message}")
            self.result = CommandResult([error], None)

    handler = Handler()
    if not handle_sigint:
        command(handler, ctx, args)
    else:
        watch_sigint(
            task=lambda: command(handler, ctx, args),
            on_sigint=lambda: handler.interrupt(),
        )
    return handler.result


def watch_sigint(
    task: Callable[[], None], on_sigint: Callable[[], None]
) -> None:
    import threading

    done = threading.Event()

    def wrapped():
        task()
        done.set()

    t = threading.Thread(target=wrapped)

    t.start()
    try:
        done.wait()
    except KeyboardInterrupt:
        print("Interrupted")
        on_sigint()
        done.wait()


#####
##### A dummy command for test purposes
#####


@dataclass
class TestCommandArgs:
    n: int
    delay: float = 1.0


def test_command(
    task: ta.TaskContext[CommandResult[list[int]]],
    exe: ta.CommandExecutionContext,
    args: TestCommandArgs,
):
    task.set_status("counting...")
    for i in range(args.n):
        time.sleep(args.delay)
        task.set_result(CommandResult([], list(range(i + 1))))
        print(i)
    task.set_status("done")
    time.sleep(args.delay)
    print("done")
