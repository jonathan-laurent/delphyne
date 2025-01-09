"""
Abstract interface and utilities for streaming tasks.
"""

import asyncio
import typing
from collections.abc import Callable
from typing import Any, Concatenate, Literal, Never, Protocol

import fastapi


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


class TaskLauncher(Protocol):
    """
    The Delphyne server can be instantiated with different task
    launchers. The simplest one is single-threaded and, which is ideal
    for debugging. We plan to add multi-threaded launchers in the
    future.
    """

    # fmt: off
    async def __call__[**P, T](
    # fmt: on
        self,
        request: fastapi.Request,
        type: type[T] | Any,
        task: StreamingTask[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> typing.AsyncGenerator[str, None]: ...
