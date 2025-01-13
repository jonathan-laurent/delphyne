"""
Delphyne Tasks.

In order to better integrate with the UI, we propose a standard
interface for Delphyne tasks, which allows streaming updates.
"""

import asyncio
import inspect
import typing
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Concatenate, Literal, Never, Protocol

import delphyne.analysis as analysis
import delphyne.analysis.feedback as fb
import delphyne.stdlib.tasks as ta

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


@dataclass
class CommandExecutionContext:
    base: analysis.DemoExecutionContext
    demo_files: list[Path]
    refresh_rate: float | None = None


@dataclass
class CommandResult[T]:
    diagnostics: list[fb.Diagnostic]
    result: T


class Command[A, T](Protocol):
    async def __call__(
        self,
        task: ta.TaskContext[CommandResult[T]],
        exe: CommandExecutionContext,
        args: A,
    ) -> None: ...


def command_args_type(cmd: Callable[..., Any]) -> Any:
    sig = inspect.signature(cmd)
    parameters = list(sig.parameters.keys())
    assert len(parameters) == 3
    hints = typing.get_type_hints(cmd)
    return hints[parameters[2]]


#####
##### A dummy command for test purposes
#####


@dataclass
class TestCommandArgs:
    n: int


async def test_command(
    task: ta.TaskContext[CommandResult[list[int]]],
    exe: ta.CommandExecutionContext,
    args: TestCommandArgs,
):
    await task.set_status("counting...")
    for i in range(args.n):
        await asyncio.sleep(0.5)
        await task.set_result(CommandResult([], list(range(i + 1))))
        print(i)
    await task.set_status("done")
    await asyncio.sleep(0.1)
    print("done")
