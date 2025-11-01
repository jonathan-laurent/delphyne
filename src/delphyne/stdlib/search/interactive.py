"""
A standard strategy for creating conversational agents.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal, overload

import delphyne.core_and_base as dp
from delphyne.stdlib import models as md
from delphyne.stdlib.nodes import Branch, branch, run
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
            useful for backpropagating `BetterValue` feedback messages.
        inner_policy_type: Ambient inner policy type. This information
            is not used at runtime but it can be provided to help type
            inference when necessary.

    !!! note
        This strategy issues normal `Branch` nodes for calls to `step`
        but `Run` nodes for calls to `process` and tool calls. Thus, if
        using `dfs` as a policy for example, the `max_depth` setting
        corresponds to the number of conversation rounds (or one plus
        the number of feedback cycles).

    !!! note "Generated feedback"
        If `produce_feedback` is `True`, this strategy produces two
        kinds of feedback sources and two backpropagation handlers.
        Feedback sources are emitted for every processing error,
        targetting the last message in the conversation (for the source
        with tag "last") or the first one (for the source with tag
        "first"). Backpropagation handlers are also tagged with "last"
        and "first" respectively, depending on whether `GoodValue` and
        `BadValue` messages should be sent to the last answer in the
        conversation or the very first one (as `BetterValue` or
        `WrongValueAlso` messages).
    """

    prefix: dp.AnswerPrefix = []
    stats = InteractStats(num_rejected=0, num_tool_call_rounds=0)
    init_resp_ref = None

    while True:
        # We ask for a response, providing the full chat history.
        resp, resp_ref = yield from branch(
            step(prefix, stats), return_ref=True
        )
        if init_resp_ref is None:
            init_resp_ref = resp_ref
        prefix.append(dp.OracleMessage("oracle", resp.answer))

        # Case where a tool call is requested.
        if isinstance(resp.parsed, dp.ToolRequests):
            tc = resp.parsed.tool_calls
            for i, t in enumerate(tc):
                assert tools is not None
                tres = yield from run(tools[type(t)](t))
                msg = dp.ToolResult(
                    "tool",
                    resp.answer.tool_calls[i],
                    t.render_result(tres),
                )
                prefix.append(msg)
            stats.num_tool_call_rounds += 1
            continue

        ans = resp.parsed.final

        # Case where the answer does not parse.
        if isinstance(ans, WrappedParseError):
            msg = dp.FeedbackMessage(
                kind="feedback",
                label=ans.error.label,
                description=ans.error.description,
                meta=ans.error.meta,
            )
            stats.num_rejected += 1
            prefix.append(msg)
            continue

        res = yield from run(process(ans, stats))

        # Case where the parsed answer cannot be processed.
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
                yield from dp.feedback("last", [err_f])
                err_f = dp.send(dp.BadValueAlso(resp, res), init_resp_ref)
                yield from dp.feedback("first", [err_f])
            continue

        # Case where we have a final good answer.
        # We define a backward feedback function and then return the answer.

        def backward(shortcut: bool, msg: dp.ValueFeedback[B]):
            # The `shortcut` argument determines whether feedback should
            # be propagated to the last response in the conversation or
            # to the first one.
            if isinstance(msg, dp.GoodValue):
                if shortcut:
                    yield dp.send(dp.BetterValue(resp), init_resp_ref)
                else:
                    yield dp.send(msg, resp_ref)
            elif isinstance(msg, dp.BadValue):
                if shortcut:
                    yield dp.send(msg, resp_ref)
                else:
                    yield dp.send(dp.BadValueAlso(resp, msg.error), resp_ref)
            elif isinstance(msg, dp.BetterValue) and unprocess:
                if (v := unprocess(msg.value)) is not None:
                    better = dp.BetterValue(dp.Response.pure(v))
                    yield dp.send(better, init_resp_ref)
            elif isinstance(msg, dp.BadValueAlso) and unprocess:
                if (v := unprocess(msg.value)) is not None:
                    bad = dp.BadValueAlso(dp.Response.pure(v), msg.error)
                    yield dp.send(bad, init_resp_ref)

        if produce_feedback:
            yield from dp.backward("last", res, partial(backward, False))
            yield from dp.backward("first", res, partial(backward, True))
        return res
