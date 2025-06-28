from collections.abc import Callable, Mapping
from typing import Any

import delphyne.core as dp
import delphyne.stdlib.queries as dq
from delphyne.stdlib import models as md
from delphyne.stdlib.nodes import Branch, branch
from delphyne.stdlib.search.dfs import dfs
from delphyne.stdlib.strategies import strategy


@strategy(name="interact")
def _interact[P, A, B, T: md.AbstractTool[Any]](
    step: Callable[
        [dp.AnswerPrefix], dp.OpaqueSpaceBuilder[P, dq.Response[A, T]]
    ],
    process: Callable[[A], dp.OpaqueSpaceBuilder[P, B | dp.Error]],
    tools: Mapping[type[T], Callable[[Any], dp.OpaqueSpaceBuilder[P, Any]]]
    | None,
) -> dp.Strategy[Branch, P, B]:
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


def interact[P, A, B, T: md.AbstractTool[Any]](
    step: Callable[
        [dp.AnswerPrefix], dp.OpaqueSpaceBuilder[P, dq.Response[A, T]]
    ],
    process: Callable[[A], dp.OpaqueSpaceBuilder[P, B | dp.Error]],
    tools: Mapping[type[T], Callable[[Any], dp.OpaqueSpaceBuilder[P, object]]]
    | None = None,
    inner_policy_type: type[P] | None = None,
) -> dp.OpaqueSpaceBuilder[P, B]:
    """
    Note: the `meta` field of `dp.Error` must be serializable.
    """

    def policy(inner_policy: P):
        return (dfs(max_branching=1), inner_policy)

    return _interact(step, process, tools).using(policy)
