from collections.abc import Callable, Mapping
from typing import Any

import delphyne.core as dp
import delphyne.stdlib.queries as dq
from delphyne.stdlib import models as md
from delphyne.stdlib.nodes import Branch, branch
from delphyne.stdlib.strategies import strategy


@strategy
def interact[P, A, B, T: md.AbstractTool[Any]](
    step: Callable[
        [dp.AnswerPrefix], dp.OpaqueSpaceBuilder[P, dq.Response[A, T]]
    ],
    process: Callable[[A], dp.OpaqueSpaceBuilder[P, B | dp.Error]],
    tools: Mapping[type[T], Callable[[Any], dp.OpaqueSpaceBuilder[P, Any]]]
    | None = None,
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch, P, B]:
    """
    Note: the `meta` field of `dp.Error` must be serializable.
    """

    prefix: dp.AnswerPrefix = []
    while True:
        resp = yield from branch(step(prefix))
        prefix += [dp.OracleMessage("oracle", resp.answer)]
        match resp.parsed:
            case dq.FinalAnswer(a):
                res = yield from branch(process(a))
                if isinstance(res, dp.Error):
                    assert res.label
                    msg = dp.FeedbackMessage(
                        "feedback", res.label, res.description, res.meta
                    )
                    prefix += [msg]
                else:
                    return res
            case dq.ToolRequests(tc):
                for i, t in enumerate(tc):
                    assert tools is not None
                    tres = yield from branch(tools[type(t)](t))
                    msg = dp.ToolResult(
                        "tool",
                        resp.answer.tool_calls[i],
                        t.render_result(tres),
                    )
                    prefix += [msg]
