"""
Utility functions for processing feedback.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, assert_type

import delphyne.core_and_base as dp
from delphyne.core.irefs import AnswerId, NodeId, SpaceId


@dataclass
class QueryFeedback:
    query: dp.AbstractQuery[Any]
    space_id: dp.irefs.SpaceId
    answer: dp.Answer
    feedback: dp.ValueFeedback[Any]


class FeedbackFilter(Protocol):
    def __call__(self, *, label: str, node_id: NodeId) -> bool: ...


def process_feedback(
    resolver: dp.IRefResolver,
    *,
    roots: Sequence[NodeId],
    filter_sources: FeedbackFilter | None = None,
    filter_backprop_handlers: FeedbackFilter | None = None,
) -> Iterable[QueryFeedback]:
    """
    Propagate feedback within a trace.

    Arguments:
        roots: Identifiers of success nodes to which `GoodValue`
            messages should be sent initially.
        filter_sources: A filter that determines which feedback sources
            (i.e. `ThrowFeedback` nodes) to activate. If `None`, no
            source is activated.
        filter_backprop_handlers: A filter that determines which
            backpropagation handlers (i.e. `BackpropagateFeedback`) to
            activate. If `None`, defaults handlers are used everywhere.
    """

    # Success nodes that have received a `GoodValue` message already. We
    # keep track of those so as to not send a message several times.
    messaged_success_nodes: set[NodeId] = set()

    def find_backprop_handlers(
        node_id: NodeId,
    ) -> Iterable[dp.BackpropagateFeedback]:
        # Starting at a success node, find all feedback backpropagation
        # handlers in the tree.
        if filter_backprop_handlers is None:
            return
        node = resolver.resolve_node(node_id)
        assert isinstance(node.node, dp.Success)
        while True:
            origin = resolver.trace.nodes[node_id]
            if isinstance(origin, dp.irefs.NestedIn):
                break
            assert_type(origin, dp.irefs.ChildOf)
            node_id = origin.node
            node = resolver.resolve_node(node_id)
            if isinstance(node.node, dp.BackpropagateFeedback):
                if filter_backprop_handlers(
                    label=node.node.label, node_id=node_id
                ):
                    yield node.node

    def propagate_good_value_message(
        node_id: NodeId,
    ) -> Iterable[QueryFeedback]:
        # For every action, we look at the space elements that it
        # contains. If the element is an answer, we send a message to
        # it. If it is a nested tree, we send a message to the
        # corresponding success node.
        node = resolver.resolve_node(node_id)
        assert isinstance(node.node, dp.Success)
        while True:
            origin = resolver.trace.nodes[node_id]
            if isinstance(origin, dp.irefs.NestedIn):
                break
            assert_type(origin, dp.irefs.ChildOf)
            elts = resolver.trace.space_elements_in_value_ref(origin.action)
            # We do not use sets, as we want to preserve order.
            elts_deduplicated = {elt: None for elt in elts}
            for elt in elts_deduplicated:
                if isinstance(elt.element, AnswerId):
                    yield from send_to_answer(
                        answer_id=elt.element,
                        space_id=elt.space,
                        message=dp.GoodValue(),
                    )
                else:
                    assert_type(elt.element, NodeId)
                    yield from propagate_good_value_message(elt.element)

    def send_to_answer(
        answer_id: AnswerId,
        space_id: SpaceId,
        message: dp.ValueFeedback[Any],
    ) -> Iterable[QueryFeedback]:
        # Send a message to an answer, yielding a feedback item.
        space = resolver.resolve_space(space_id)
        ans = resolver.resolve_answer(answer_id)
        assert space != "main"
        source = space.source()
        assert isinstance(source, dp.AttachedQuery)
        query = source.query
        yield QueryFeedback(
            query=query, space_id=space_id, answer=ans, feedback=message
        )

    def send_to_success_node(
        node_id: NodeId, message: dp.ValueFeedback[Any]
    ) -> Iterable[QueryFeedback]:
        # We do not process a GoodValue message multiple times at a
        # given success node.
        if isinstance(message, dp.GoodValue):
            if node_id in messaged_success_nodes:
                return
            messaged_success_nodes.add(node_id)
        handlers = list(find_backprop_handlers(node_id))
        if not handlers:
            yield from propagate_good_value_message(node_id)
            return
        for handler in handlers:
            for attached in handler.back(message):
                full_ref = attached.dst.ref
                _dst = resolver.trace.convert_global_space_path(full_ref)
                assert False

    if roots:
        for root in roots:
            yield from send_to_success_node(root, dp.GoodValue())
