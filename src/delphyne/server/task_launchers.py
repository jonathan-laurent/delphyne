"""
Abstract interface and utilities for streaming tasks.
"""

import typing
from typing import Any, Protocol

import fastapi

from delphyne.stdlib.tasks import StreamingTask


class TaskLauncher(Protocol):
    """
    The Delphyne server can be instantiated with different task
    launchers. The simplest one is single-threaded and, which is ideal
    for debugging. We plan to add multi-threaded launchers in the
    future.
    """

    # fmt: off
    def __call__[**P, T](
    # fmt: on
        self,
        request: fastapi.Request,
        type: type[T] | Any,
        task: StreamingTask[P, T],
        *args: P.args,
        **kwargs: P.kwargs,
    ) -> typing.AsyncGenerator[str, None]: ...
