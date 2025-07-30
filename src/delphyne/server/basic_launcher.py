"""
A single-threaded task launcher for debugging purposes.
"""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any

import pydantic
from fastapi import Request

from delphyne.server.task_launchers import StreamingTask, TaskLauncher
from delphyne.stdlib import tasks
from delphyne.stdlib.tasks import TaskMessage

_CONNECTION_POLLING_TIME = 0.5
"""
The time interval (in seconds) at which the HTTP connection is polled so
that the task can be cancelled if the connection breaks.
"""


class _BasicTaskContext[T](tasks.TaskContext[T]):
    def __init__(self):
        self.messages_queue = asyncio.Queue[TaskMessage[T]]()
        self._interrupted = asyncio.Event()

    def interrupt(self):
        return self._interrupted.set()

    def interruption_requested(self) -> bool:
        return self._interrupted.is_set()

    def log(self, message: str) -> None:
        self.messages_queue.put_nowait(("log", message))

    async def set_status(self, message: str) -> None:
        self.messages_queue.put_nowait(("set_status", message))
        await asyncio.sleep(0)

    async def set_result(self, result: T) -> None:
        self.messages_queue.put_nowait(("set_result", result))
        await asyncio.sleep(0)

    async def raise_internal_error(self, message: str) -> None:
        self.messages_queue.put_nowait(("internal_error", message))
        await asyncio.sleep(0)


class BasicLauncher(TaskLauncher):
    async def __call__[**P, T](
        self,
        request: Request | None,
        type: type[T] | Any,
        task: StreamingTask[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> AsyncGenerator[str, None]:  # fmt: skip
        context = _BasicTaskContext[T]()
        background = asyncio.create_task(task(context, *args, **kwargs))
        adapter = pydantic.TypeAdapter[TaskMessage[type]](TaskMessage[type])
        try:
            while not context.messages_queue.empty() or not background.done():
                if request is not None and await request.is_disconnected():
                    break
                try:
                    message = await asyncio.wait_for(
                        context.messages_queue.get(),
                        timeout=_CONNECTION_POLLING_TIME,
                    )
                    yield json.dumps(adapter.dump_python(message)) + "\n\n"
                except asyncio.TimeoutError:
                    pass
        finally:
            background.cancel()
        if (exn := background.exception()) is not None and request is None:
            raise exn
