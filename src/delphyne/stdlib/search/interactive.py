from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

import delphyne.core as dp
import delphyne.stdlib.queries as dq
from delphyne.stdlib import models as md
from delphyne.stdlib.nodes import Branch, branch


@dataclass(frozen=True)
class InteractStats:
    num_rejected: int
    num_tool_calls: int


def interact[P, A, B, T: md.AbstractTool[Any]](
    step: Callable[
        [dp.AnswerPrefix, InteractStats],
        dp.Opaque[P, dq.Response[A, T]],
    ],
    process: Callable[[A, InteractStats], dp.Opaque[P, B | dp.Error]],
    tools: Mapping[type[T], Callable[[Any], dp.Opaque[P, Any]]] | None = None,
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch, P, B]:
    """
    Note: the `meta` field of `dp.Error` must be serializable.
    """

    prefix: dp.AnswerPrefix = []
    stats = InteractStats(num_rejected=0, num_tool_calls=0)
    while True:
        resp = yield from branch(step(prefix, stats))
        prefix += [dp.OracleMessage("oracle", resp.answer)]
        match resp.parsed:
            case dq.FinalAnswer(a):
                res = yield from branch(process(a, stats))
                if isinstance(res, dp.Error):
                    assert res.label
                    msg = dp.FeedbackMessage(
                        "feedback", res.label, res.description, res.meta
                    )
                    stats = replace(stats, num_rejected=stats.num_rejected + 1)
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
                    stats = replace(
                        stats, num_tool_calls=stats.num_tool_calls + 1
                    )
                    prefix += [msg]
