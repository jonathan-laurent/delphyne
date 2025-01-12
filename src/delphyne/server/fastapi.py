"""
FastAPI Application.
"""

from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.requests import Request
from fastapi.responses import StreamingResponse

import delphyne.analysis as analysis
import delphyne.analysis.feedback as fb
import delphyne.core as dp
import delphyne.server.commands as cm
import delphyne.server.tasks as ta
import delphyne.stdlib as std


def make_server(launcher: ta.TaskLauncher):
    app = FastAPI()

    @app.get("/ping-delphyne")
    def _():
        return "OK"

    @app.post("/demo-feedback")
    def _(
        request: Request,
        # Strangely: `demo: dp.Demo` does not work with fastapi...
        demo: dp.QueryDemo | dp.StrategyDemo,
        context: analysis.DemoExecutionContext,
    ):
        stream_eval = ta.stream_of_fun(analysis.evaluate_demo)
        extra = std.stdlib_globals()
        stream = launcher(
            request, fb.DemoFeedback, stream_eval, demo, context, extra
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    @app.post("/count")
    async def _(request: Request, n: Annotated[int, Body(embed=True)]):
        stream = launcher(request, int, ta.counting_generator, n)
        return StreamingResponse(stream, media_type="text/event-stream")

    @app.post("/execute-command")
    async def _(
        request: Request,
        spec: cm.CommandSpec,
        context: cm.CommandExecutionContext,
    ):
        stream = launcher(
            request, cm.CommandResult, cm.execute_command, context, spec
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    return app
