"""
FastAPI Application.
"""

from pathlib import Path
from typing import Annotated

from fastapi import Body, FastAPI
from fastapi.requests import Request
from fastapi.responses import StreamingResponse

import delphyne.analysis as analysis
import delphyne.analysis.feedback as fb
import delphyne.core_and_base as dp
import delphyne.server.execute_command as cm
import delphyne.stdlib as std
import delphyne.stdlib.tasks as ta
from delphyne.server.task_launchers import TaskLauncher


def make_server(launcher: TaskLauncher):
    app = FastAPI()

    @app.get("/ping-delphyne")
    def _():
        return "OK"

    @app.post("/demo-feedback")
    def _(
        request: Request,
        # Strangely: `demo: dp.Demo` does not work with fastapi...
        demo: dp.QueryDemo | dp.StrategyDemo,
        context: std.ExecutionContext,
        workspace_root: Annotated[str, Body(embed=True)],
    ):
        context = context.with_root(Path(workspace_root))
        stream = launcher(
            request,
            fb.DemoFeedback,
            ta.stream_of_fun(analysis.safe_evaluate_demo),
            demo=demo,
            object_loader=(
                lambda: context.object_loader(
                    extra_objects=std.stdlib_globals()
                )
            ),
            answer_database_loader=(
                lambda loader: dp.standard_answer_loader(
                    Path(workspace_root), loader
                )
            ),
            implicit_answer_generators=lambda loader: (
                std.stdlib_implicit_answer_generators(context.data_dirs)
            ),
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
        context: std.ExecutionContext,
        workspace_root: Annotated[str, Body(embed=True)],
    ):
        stream = launcher(
            request,
            ta.CommandResult,
            cm.execute_command,
            context,
            Path(workspace_root),
            spec,
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    return app
