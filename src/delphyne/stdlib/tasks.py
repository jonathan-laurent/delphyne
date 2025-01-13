"""
Delphyne Tasks.

In order to better integrate with the UI, we propose a standard
interface for Delphyne tasks, which allows streaming updates.
"""

import asyncio
import inspect
import os
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

#####
##### Tasks
#####


class TaskContext[T](typing.Protocol):
    def log(self, message: str) -> None: ...
    async def set_status(self, message: str) -> None: ...
    async def set_result(self, result: T) -> None: ...
    async def raise_internal_error(self, message: str) -> None: ...


type StreamingTask[**P, T] = Callable[
    Concatenate[TaskContext[T], P],
    typing.Coroutine[None, None, None],
]


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

    async def stream(
        context: TaskContext[T], *args: P.args, **kwargs: P.kwargs
    ):
        try:
            ret = f(*args, **kwargs)
            await context.set_result(ret)
        except Exception as e:
            await context.raise_internal_error(repr(e))

    return stream


async def counting_generator(task: TaskContext[Never], n: int):
    """
    A basic task example, to be used for testing.
    """
    try:
        for i in range(n):
            await asyncio.sleep(1)
            print(f"Send: {i}", flush=True)
            await task.set_status(str(i))
            task.log(f"Sent: {i}")
            i += 1
    finally:
        print("Generator over.", flush=True)


#####
##### Particular Case: Commands
#####


@dataclass(frozen=True)
class CommandExecutionContext:
    base: analysis.DemoExecutionContext
    demo_files: Sequence[Path]
    refresh_rate: float | None = None


@dataclass(frozen=True)
class CommandResult[T]:
    diagnostics: Sequence[fb.Diagnostic]
    result: T


class Command[A, T](Protocol):
    async def __call__(
        self,
        task: ta.TaskContext[CommandResult[T]],
        exe: CommandExecutionContext,
        args: A,
    ) -> None: ...


def command_args_type(cmd: Command[Any, Any]) -> Any:
    sig = inspect.signature(cmd)
    parameters = list(sig.parameters.keys())
    assert len(parameters) == 3
    hints = typing.get_type_hints(cmd)
    return hints[parameters[2]]


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


def command_name(cmd: Command[Any, Any]) -> str:
    return cmd.__name__  # type: ignore


def run_command[A, T](
    command: Command[A, T],
    args: A,
    ctx: CommandExecutionContext,
    dump_statuses: Path | None = None,
    dump_result: Path | None = None,
    dump_log: Path | None = None,
    add_header: bool = True,
) -> CommandResult[T | None]:
    """
    A simple way to run a command, blocking and dumping logs on disk.

    Intermediate directories are created if necessary.
    """

    class Handler:
        def __init__(self):
            self.result: CommandResult[T | None] = CommandResult([], None)

        def log(self, message: str) -> None:
            if dump_log is not None:
                os.makedirs(dump_log.parent, exist_ok=True)
                with open(dump_log, "a") as f:
                    f.write(message + "\n")

        async def set_status(self, message: str) -> None:
            if dump_statuses is not None:
                os.makedirs(dump_statuses.parent, exist_ok=True)
                with open(dump_statuses, "a") as f:
                    f.write(message + "\n")

        async def set_result(self, result: CommandResult[T]) -> None:
            self.result = result
            if dump_result is not None:
                ret_ty = command_result_type(command)
                ret_ty = CommandResult[ret_ty | None]
                ret: Any = ty.pydantic_dump(ret_ty, result)
                if add_header:
                    args_type = command_args_type(command)
                    ret = {
                        "command": command_name(command),
                        "args": ty.pydantic_dump(args_type, args),
                        "outcome": ret,
                    }
                os.makedirs(dump_result.parent, exist_ok=True)
                with open(dump_result, "w") as f:
                    f.write(pretty_yaml(ret))

        async def raise_internal_error(self, message: str) -> None:
            error = ("error", f"Internal error: {message}")
            self.result = CommandResult([error], None)

    handler = Handler()
    asyncio.run(command(handler, ctx, args))
    return handler.result


#####
##### A dummy command for test purposes
#####


@dataclass
class TestCommandArgs:
    n: int
    delay: float = 1.0


async def test_command(
    task: ta.TaskContext[CommandResult[list[int]]],
    exe: ta.CommandExecutionContext,
    args: TestCommandArgs,
):
    await task.set_status("counting...")
    for i in range(args.n):
        await asyncio.sleep(args.delay)
        await task.set_result(CommandResult([], list(range(i + 1))))
        print(i)
    await task.set_status("done")
    await asyncio.sleep(args.delay)
    print("done")
