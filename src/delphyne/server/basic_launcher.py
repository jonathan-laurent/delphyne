"""
A single-threaded task launcher for debugging purposes.
"""

import queue
import threading
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
        self.messages_queue = queue.Queue[TaskMessage[T]]()
        self._interrupted = threading.Event()

    def interrupt(self):
        return self._interrupted.set()

    def interruption_requested(self) -> bool:
        return self._interrupted.is_set()

    def log(self, message: str) -> None:
        self.messages_queue.put_nowait(("log", message))

    def set_status(self, message: str) -> None:
        self.messages_queue.put_nowait(("set_status", message))

    def set_result(self, result: T) -> None:
        self.messages_queue.put_nowait(("set_result", result))

    def raise_internal_error(self, message: str) -> None:
        self.messages_queue.put_nowait(("internal_error", message))

    def set_pull_status(self, pull: tasks.PullStatusFn) -> None:
        pass

    def set_pull_result(self, pull: tasks.PullResultFn[T]) -> None:
        pass


class BasicLauncher(TaskLauncher):
    async def __call__[**P, T](
        self,
        request: Request | None,
        type: type[T] | Any,
        task: StreamingTask[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> AsyncGenerator[str, None]:  # fmt: skip
        import asyncio

        context = _BasicTaskContext[T]()
        background = asyncio.create_task(
            asyncio.to_thread(lambda: task(context, *args, **kwargs))
        )
        adapter = pydantic.TypeAdapter[TaskMessage[type]](TaskMessage[type])
        try:
            while not context.messages_queue.empty() or not background.done():
                if request is not None and await request.is_disconnected():
                    break
                if not context.messages_queue.empty():
                    message = context.messages_queue.get_nowait()
                    yield adapter.dump_json(message).decode() + "\n\n"
                else:
                    await asyncio.sleep(_CONNECTION_POLLING_TIME)
        finally:
            context.interrupt()
        if (exn := background.exception()) is not None and request is None:
            raise exn
