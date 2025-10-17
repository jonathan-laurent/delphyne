"""
A standard strategy for creating conversational agents.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, overload

import delphyne.core_and_base as dp
from delphyne.stdlib import models as md
from delphyne.stdlib.nodes import Branch, branch
from delphyne.stdlib.opaque import Opaque
from delphyne.stdlib.queries import WrappedParseError


@dataclass
class InteractStats:
    """
    Statistics maintained by `interact`.

    Attributes:
        num_rejected: Number of answers that have been rejected so far,
            due to either parsing or processing errors.
        num_tool_call_rounds: Number of tool call rounds that have been
            reuqetsed by the LLM so far (a round consists in a single
            message that can contain several tool call requests).
    """

    num_rejected: int
    num_tool_call_rounds: int


@overload
def interact[P, A, B, T: md.AbstractTool[Any]](
    step: Callable[
        [dp.AnswerPrefix, InteractStats],
        Opaque[P, dp.Response[A | WrappedParseError, T]],
    ],
    *,
    process: Callable[[A, InteractStats], Opaque[P, B | dp.Error]],
    tools: Mapping[type[T], Callable[[Any], Opaque[P, Any]]] | None = None,
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch, P, B]: ...


@overload
def interact[P, A, B, T: md.AbstractTool[Any]](
    step: Callable[
        [dp.AnswerPrefix, InteractStats],
        Opaque[P, dp.Response[A | WrappedParseError, T]],
    ],
    *,
    process: Callable[[A, InteractStats], Opaque[P, B | dp.Error]],
    tools: Mapping[type[T], Callable[[Any], Opaque[P, Any]]] | None = None,
    produce_feedback: Literal[True] = True,
    unprocess: Callable[[B], A | None] | None = None,
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch | dp.Feedback, P, B]: ...


def interact[P, A, B, T: md.AbstractTool[Any]](
    step: Callable[
        [dp.AnswerPrefix, InteractStats],
        Opaque[P, dp.Response[A | WrappedParseError, T]],
    ],
    *,
    process: Callable[[A, InteractStats], Opaque[P, B | dp.Error]],
    tools: Mapping[type[T], Callable[[Any], Opaque[P, Any]]] | None = None,
    produce_feedback: bool = False,
    unprocess: Callable[[B], A | None] | None = None,
    inner_policy_type: type[P] | None = None,
) -> dp.Strategy[Branch | dp.Feedback, P, B]:
    """
    A standard strategy for creating conversational agents.

    A common pattern for interacting with LLMs is to have multi-message
    exchanges where the full conversation history is resent repeatedly.
    LLMs are also often allowed to request tool calls. This strategy
    implements this pattern. It is meant to be inlined into a wrapping
    strategy (since it is not decorated with `strategy`).

    Parameters:
        step: A parametric opaque space, induced by a strategy or query
            that takes as an argument the current chat history (possibly
            empty) along with some statistics, and returns an answer to
            be processed. Oftentimes, this parametric opaque space is
            induced by a query with a special `prefix` field for
            receiving the chat history (see `Query`).
        process: An opaque space induced by a query or strategy that is
            called on all model responses that are not tool calls, and
            which returns either a final response to be returned, or an
            error to be transmitted to the model as feedback (as an
            `Error` value with an absent or serializable `meta` field).
        tools: A mapping from supported tool interfaces to
            implementations. Tools themselves can be implemented using
            arbitrary strategies or queries, allowing the integration of
            horizontal and vertical LLM pipelines.
        produce_feedback: Whether or not to produce `Feedback` nodes.
        unprocess: If `produce_feedback` is `True`, this function is
            useful for backpropagating `Shortcut` feedback messages.
        inner_policy_type: Ambient inner policy type. This information
            is not used at runtime but it can be provided to help type
            inference when necessary.
    """

    prefix: dp.AnswerPrefix = []
    stats = InteractStats(num_rejected=0, num_tool_call_rounds=0)
    init_resp_ref = None
    while True:
        resp, resp_ref = yield from branch(
            step(prefix, stats), return_space_ref=True
        )
        if init_resp_ref is None:
            init_resp_ref = resp_ref
        prefix.append(dp.OracleMessage("oracle", resp.answer))
        match resp.parsed:
            case dp.FinalAnswer(a):
                if isinstance(a, WrappedParseError):
                    msg = dp.FeedbackMessage(
                        kind="feedback",
                        label=a.error.label,
                        description=a.error.description,
                        meta=a.error.meta,
                    )
                    stats.num_rejected += 1
                    prefix.append(msg)
                else:
                    res = yield from branch(process(a, stats))
                    if isinstance(res, dp.Error):
                        msg = dp.FeedbackMessage(
                            kind="feedback",
                            label=res.label,
                            description=res.description,
                            meta=res.meta,
                        )
                        stats.num_rejected += 1
                        prefix.append(msg)
                        if produce_feedback:
                            err_f = dp.send(dp.BadValue(res), resp_ref)
                            yield from dp.feedback("error", [err_f])
                    else:
                        if produce_feedback:

                            def backward(msg: dp.HindsightMessage[B]):
                                if isinstance(msg, dp.GoodValue):
                                    yield dp.send(msg, resp_ref)
                                    yield dp.send(
                                        dp.Shortcut(resp), init_resp_ref
                                    )
                                elif (
                                    isinstance(msg, dp.Shortcut) and unprocess
                                ):
                                    # We send a shortcut to the initial
                                    # response, knowing that the raw
                                    # answer will be discarded and only
                                    # the parsed answer will matter.
                                    v = unprocess(msg.value)
                                    if v is not None:
                                        yield dp.send(
                                            dp.Shortcut(
                                                dp.Response[A, T](
                                                    resp.answer,  # irrelevant
                                                    dp.FinalAnswer(v),
                                                )
                                            ),
                                            init_resp_ref,
                                        )
                                elif isinstance(msg, dp.BadValue):
                                    yield dp.send(msg, resp_ref)

                            dp.backward("shortcut", res, backward)
                        return res
            case dp.ToolRequests(tc):
                for i, t in enumerate(tc):
                    assert tools is not None
                    tres = yield from branch(tools[type(t)](t))
                    msg = dp.ToolResult(
                        "tool",
                        resp.answer.tool_calls[i],
                        t.render_result(tres),
                    )
                    prefix.append(msg)
                stats.num_tool_call_rounds += 1
