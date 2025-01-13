"""
Standard commands for answering queries.
"""

from dataclasses import dataclass

import delphyne.analysis as analysis
import delphyne.core as dp
import delphyne.stdlib as std
import delphyne.stdlib.models as mo
import delphyne.stdlib.queries as qu
import delphyne.stdlib.tasks as ta

DEFAULT_OPENAI_MODEL = "gpt-4o"


@dataclass
class AnswerQueryArgs:
    query: str
    completions: int
    prompt_only: bool
    params: dict[str, object]
    options: mo.RequestOptions
    args: dict[str, object]


@dataclass
class AnswerQueryResponse:
    prompt: mo.Chat | None
    response: str | None


async def answer_query(
    task: ta.TaskContext[ta.CommandResult[AnswerQueryResponse]],
    exe: ta.CommandExecutionContext,
    cmd: AnswerQueryArgs,
):
    # TODO: no examples for now. Also, we have to externalize this anyway.
    loader = analysis.ObjectLoader(exe.base)
    query = loader.load_query(cmd.query, cmd.args)
    env = dp.TemplatesManager(exe.base.strategy_dirs)
    prompt = qu.create_prompt(query, [], cmd.params, env)
    await task.set_result(
        ta.CommandResult([], AnswerQueryResponse(prompt, None))
    )
    if cmd.prompt_only:
        return
    answer = ""
    model = std.openai_model(DEFAULT_OPENAI_MODEL)
    async for chunk in model.stream_request(prompt, cmd.options):
        answer += chunk
        await task.set_result(
            ta.CommandResult([], AnswerQueryResponse(prompt, answer))
        )
