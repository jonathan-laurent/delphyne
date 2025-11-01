"""
Utility functions for processing feedback.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, assert_type

import delphyne.analysis as an
import delphyne.core as dp
import delphyne.stdlib.hindsight_feedback as hf
from delphyne.core.irefs import AnswerId, NodeId, SpaceId

#####
##### General propagation mechanism
#####


@dataclass
class AnswerFeedback:
    query: dp.AbstractQuery[Any]
    space_id: dp.irefs.SpaceId
    answer_id: AnswerId
    answer: dp.Answer
    feedback: hf.ValueFeedback[Any]


class FeedbackFilter(Protocol):
    def __call__(self, *, label: str, node_id: NodeId) -> bool: ...


def process_feedback(
    resolver: an.IRefResolver,
    *,
    roots: Sequence[NodeId],
    filter_sources: FeedbackFilter | None = None,
    filter_backprop_handlers: FeedbackFilter | None = None,
    verbose: bool = True,
) -> Iterable[AnswerFeedback]:
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

    def log(msg: str):
        if verbose:
            print(msg)

    def find_backprop_handlers(
        node_id: NodeId,
    ) -> Iterable[hf.BackpropagateFeedback]:
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
            if isinstance(node.node, hf.BackpropagateFeedback):
                if filter_backprop_handlers(
                    label=node.node.label, node_id=node_id
                ):
                    yield node.node

    def propagate_good_value_message(
        node_id: NodeId,
    ) -> Iterable[AnswerFeedback]:
        # For every action, we look at the space elements that it
        # contains. If the element is an answer, we send a message to
        # it. If it is a nested tree, we send a message to the
        # corresponding success node.
        log(f"Auto-propagating GoodValue from node {node_id}.")
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
                        message=hf.GoodValue(),
                    )
                else:
                    assert_type(elt.element, NodeId)
                    yield from propagate_good_value_message(elt.element)
            node_id = origin.node

    def send_to_answer(
        answer_id: AnswerId,
        space_id: SpaceId,
        message: hf.ValueFeedback[Any],
    ) -> Iterable[AnswerFeedback]:
        # log(f"Sending {type(message).__name__} to answer {answer_id}.")
        # Send a message to an answer, yielding a feedback item.
        space = resolver.resolve_space(space_id)
        ans = resolver.resolve_answer(answer_id)
        assert space != "main"
        source = space.source()
        assert isinstance(source, dp.AttachedQuery)
        query = source.query
        yield AnswerFeedback(
            query=query,
            space_id=space_id,
            answer_id=answer_id,
            answer=ans,
            feedback=message,
        )

    def send_to_success_node(
        node_id: NodeId, message: hf.ValueFeedback[Any]
    ) -> Iterable[AnswerFeedback]:
        log(f"Sending {type(message).__name__} to success node {node_id}.")
        # We do not process a GoodValue message multiple times at a
        # given success node.
        if isinstance(message, hf.GoodValue):
            if node_id in messaged_success_nodes:
                return
            messaged_success_nodes.add(node_id)
        handlers = list(find_backprop_handlers(node_id))
        if not handlers and isinstance(message, hf.GoodValue):
            # If the message is `GoodValue` and there is no registered
            # handler, we do the default propagation.
            yield from propagate_good_value_message(node_id)
            return
        for handler in handlers:
            for attached in handler.back(message):
                yield from send_attached_message(attached)

    def send_attached_message(
        attached: hf.AttachedFeedback[Any],
    ) -> Iterable[AnswerFeedback]:
        ref = attached.dst
        nref = resolver.trace.convert_global_node_ref(ref.node)
        eref = resolver.trace.convert_space_element_ref(nref, ref.element)
        if isinstance(eref.element, NodeId):
            yield from send_to_success_node(eref.element, attached.msg)
        else:
            assert_type(eref.element, AnswerId)
            yield from send_to_answer(
                answer_id=eref.element,
                space_id=eref.space,
                message=attached.msg,
            )

    def iter_sources() -> Iterable[tuple[NodeId, hf.ThrowFeedback]]:
        for node_id in resolver.trace.nodes:
            tree = resolver.resolve_node(node_id)
            if isinstance(tree.node, hf.ThrowFeedback):
                yield (node_id, tree.node)

    # If some roots are specified, we go through them
    if roots:
        for root in roots:
            yield from send_to_success_node(root, hf.GoodValue())

    # Then use provided sources to send messages
    if filter_sources is not None:
        for node_id, node in iter_sources():
            if filter_sources(label=node.label, node_id=node_id):
                for msg in node.messages:
                    yield from send_attached_message(msg)


#####
##### Automatically extracting examples
#####


@dataclass
class ExtractedExample:
    """
    Attributes:
        modified: Whether the returned answer has been modified from the
            original answer described by `answer_id`, via a
            `BetterValue` feedback message.
    """

    query: dp.AbstractQuery[Any]
    answer: dp.Answer
    answer_id: AnswerId
    modified: bool


def _surrounding_spaces(trace: dp.Trace, node: NodeId) -> Sequence[SpaceId]:
    """
    Return the list of consecutive space ids surrounding the current
    node, **not** including the main space.
    """
    spaces_rev: list[SpaceId] = []
    while True:
        origin = trace.nodes[node]
        if isinstance(origin, dp.irefs.NestedIn):
            space_origin = trace.spaces[origin.space]
            if isinstance(space_origin, dp.irefs.MainSpace):
                break
            spaces_rev.append(origin.space)
            node = space_origin.node
        else:
            assert_type(origin, dp.irefs.ChildOf)
            node = origin.node
    return list(reversed(spaces_rev))


def match_handler_pattern(
    pattern: str,
    surrounding_space_tags: Sequence[Sequence[dp.Tag]],
    handler_label: str,
) -> bool:
    comps = pattern.split("/")
    assert comps, "Empty tag pattern"
    if len(comps) >= 2:
        spaces_descr = comps[:-1]
        label = comps[-1]
    else:
        spaces_descr = []
        label = comps[0]
    return (
        handler_label == label
        and len(spaces_descr) == len(surrounding_space_tags)
        and all(
            all(t in tags for t in pat.split("&"))
            for pat, tags in zip(spaces_descr, surrounding_space_tags)
        )
    )


def _surrounding_spaces_tags(
    resolver: an.IRefResolver, node_id: NodeId
) -> Iterable[Sequence[dp.Tag]]:
    """
    Return the list of consecutive space tags surrounding the current
    node, **not** including the main space.
    """
    space_ids = _surrounding_spaces(resolver.trace, node_id)
    for id in space_ids:
        space = resolver.resolve_space(id)
        assert space != "main"
        yield space.tags()


def extract_examples(
    resolver: an.IRefResolver,
    *,
    roots: Sequence[NodeId],
    backprop_handler_tags: Sequence[str] | None = None,
) -> Iterable[ExtractedExample]:
    """
    Extract individual query/answer pairs using feedback propagation.
    """

    def filter_backprop_handlers(*, label: str, node_id: NodeId):
        surrounding_space_tags = list(
            _surrounding_spaces_tags(resolver, node_id)
        )
        return any(
            match_handler_pattern(pat, surrounding_space_tags, label)
            for pat in backprop_handler_tags or []
        )

    feedback_items = process_feedback(
        resolver,
        roots=roots,
        filter_sources=None,
        filter_backprop_handlers=filter_backprop_handlers,
    )
    for f in feedback_items:
        if isinstance(f.feedback, hf.BetterValue):
            answer = f.query.unparse(f.feedback.value)
            assert answer is not None, (
                f"Unable to unparse answer for {f.query.query_name()}: "
                f"{f.feedback.value}"
            )
            modified = True
        elif isinstance(f.feedback, hf.GoodValue):
            answer = f.answer
            modified = False
        else:
            continue
        yield ExtractedExample(
            query=f.query,
            answer=answer,
            answer_id=f.answer_id,
            modified=modified,
        )
