"""
FastAPI Application.
"""

from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.requests import Request
from fastapi.responses import StreamingResponse

from delphyne.core.demos import Demonstration
from delphyne.server import feedback as fb
from delphyne.server import tasks as tasks
from delphyne.server.commands import (
    CommandExecutionContext,
    CommandResult,
    CommandSpec,
    execute_command,
)
from delphyne.server.evaluate_demo import ExecutionContext, evaluate_demo


def make_server(launcher: tasks.TaskLauncher):
    app = FastAPI()

    @app.get("/ping-delphyne")
    def _():
        return "OK"

    @app.post("/demo-feedback")
    def _(request: Request, demo: Demonstration, context: ExecutionContext):
        stream_eval = tasks.stream_of_fun(evaluate_demo)
        stream = launcher(request, fb.DemoFeedback, stream_eval, demo, context)
        return StreamingResponse(stream, media_type="text/event-stream")

    @app.post("/count")
    async def _(request: Request, n: Annotated[int, Body(embed=True)]):
        stream = launcher(request, int, tasks.counting_generator, n)
        return StreamingResponse(stream, media_type="text/event-stream")

    @app.post("/execute-command")
    async def _(
        request: Request, spec: CommandSpec, context: CommandExecutionContext
    ):
        stream = launcher(
            request, CommandResult, execute_command, context, spec
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    return app
