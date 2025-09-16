"""
Type definitions for the feedback that results from evaluating a demo.
"""

from dataclasses import dataclass
from typing import Literal

#####
##### Diagnostic types
#####


type DiagnosticType = Literal["error", "warning", "info"]
"""Diagnostic type."""


type Diagnostic = tuple[DiagnosticType, str]
"""A diagnostic gathers a type (i.e. severity) and a message."""


#####
##### Browsable Trace
#####


# Defining a rich trace format that is easy to visualize and display.
# Can be skipped on first reading.


type TraceNodeId = int
"""Global node id, as set in `core.traces.Trace`."""


type TraceAnswerId = int
"""Global answer id, as set by `core.traces.Trace`."""


type TraceActionId = int
"""Index of an action within a given node."""


type TraceNodePropertyId = int
"""Index of a property within a given node. A property is an element
that can be listed in the UI, which is either an attached query, a
nested tree or some data."""


@dataclass(kw_only=True)
class ValueRepr:
    """
    Multiple representations for a Python object.

    We allow providing several representations for Python objects:
    short, one-liner string descriptions, detailed descriptions, JSON
    representation... All of these can be leveraged by different tools
    and UI components.

    Attributes:
        short: A short representation, typically obtained using the
            `str` function.
        long: A longer, often multi-line representation, typically
            obtained using the `pprint` module.
        json: A JSON representation of the object.
        json_provided: Whether a JSON representation is provided (the
            JSON field is `None` otherwise). This is not always the case
            since not all Python objects can be serialized to JSON.
    """

    short: str
    long: str | None
    json_provided: bool
    json: object


@dataclass(kw_only=True)
class Reference:
    """
    A reference to a space or to a value.

    Several human-readable representations are provided:

    Attributes:
        with_ids: A pretty-printed, id-based reference.
        with_hints: A pretty-printed, hint-based reference. These are
            typically available in the output of the demonstration
            interpreter, but not when converting arbitrary traces that
            result from running policies.
    """

    with_ids: str
    with_hints: str | None


@dataclass
class Data:
    """
    Generic property that displays some data.

    Attributes:
        kind: Always "data".
        content: string representation of the data content.
    """

    kind: Literal["data"]
    content: str


@dataclass(kw_only=True)
class NestedTree:
    """
    A nested tree.

    Attributes:
        kind: Always "nested".
        strategy: Name of the strategy function that induces the tree.
        args: Arguments passed to the strategy function.
        tags: Tags attached to the space induced by the tree.
        node_id: Identifier of the root node of the nested tree, or
            `None` if it is not in the trace (i.e., the nested tree hasn't
            been explored).
    """

    kind: Literal["nested"]
    strategy: str
    args: dict[str, ValueRepr]
    tags: list[str]
    node_id: TraceNodeId | None  # None if the subtree hasn't been explored


@dataclass(kw_only=True)
class Answer:
    """
    An answer to a query.

    Attributes:
        id: Unique answer identifier.
        value: Parsed answer value.
        hint: If the trace results from executing a demonstration (vs
            running a policy with tracing enabled), then `hint` is
            either `()` if the answer corresponds to the default answer
            and `(l,)` if the answer is labeled with `l`. Otherwise, it
            is `None`.
    """

    id: TraceAnswerId
    hint: tuple[()] | tuple[str] | None
    value: ValueRepr


@dataclass(kw_only=True)
class Query:
    """
    Information about a query.

    Attributes:
        kind: Always "query".
        name: Name of the query.
        args: Query arguments, serialized in JSON.
        tags: Tags attached to the space induced by the query.
        answers: All answers to the query present in the trace.
    """

    kind: Literal["query"]
    name: str
    args: dict[str, object]
    tags: list[str]
    answers: list[Answer]


type NodeProperty = Data | NestedTree | Query
"""Description of a node property (see `NodePropertyId`)."""


@dataclass(kw_only=True)
class Action:
    """
    An action associated with a node.

    Attributes:
        ref: Pretty-printed local reference for the action.
        hints: If the trace results from executing a demonstration,
            this provides the list of hints that can be used to recover
            the action through navigation. Otherwise, it is `None`. Note
            that this is not identical to `ref.with_hints`. Both could
            plausibly be shown in the UI but the former is more concise.
        related_success_nodes: List of related success nodes. A related
            success node is a node whose attached value was used in
            building the action. Indeed, in the VSCode extension's Path
            View, we get a sequence of actions and for each of them the
            list of success paths that were involved in building that
            action.
        related_answers: List of related answers. A related answer is an
            answer to a local query that is used in building the action.
            Storing this information is useful to detect useless answers
            that are not used in any action.
        destination: Id of the child node that the action leads to.
    """

    ref: Reference
    hints: list[str] | None
    related_success_nodes: list[TraceNodeId]
    related_answers: list[TraceAnswerId]
    value: ValueRepr
    destination: TraceNodeId


type NodeOrigin = (
    Literal["root"]
    | tuple[Literal["child"], TraceNodeId, TraceActionId]
    | tuple[Literal["nested"], TraceNodeId, TraceNodePropertyId]
)
"""
Origin of a node.

A node can be the global root, the child of another node, or the root of
a nested tree.
"""


@dataclass(kw_only=True)
class Node:
    """
    Information about a node.

    Attributes:
        kind: Name of the node type, or `Success`.
        success_value: The success value if the node is a success leaf,
            or `None` otherwise.
        summary_message: A short summary message (see the
            `Node.sumary_message` method).
        leaf_node: Whether the node is a leaf node
        label: A label describing the node, which can be useful for
            writing node selectors (although there is currently no
            guarantee that the label constitutes a valid selector
            leading to the node). Currently, the label shows all node
            tags, separated by "&".
        tags: The list of all tags attached to the node.
        properties: List of node properties (attached queries, nested
            trees, data fields...). Each property is accompanied by a
            pretty-printed, local space reference.
        actions: A list of explored actions.
        origin: The origin of the node in the global trace.
    """

    # TODO: Make node labels into valid selectors that can be used with
    # the `at` instruction in demonstration tests.

    kind: str
    success_value: ValueRepr | None
    summary_message: str | None
    leaf_node: bool
    label: str | None
    tags: list[str]
    properties: list[tuple[Reference, NodeProperty]]
    actions: list[Action]
    origin: NodeOrigin


@dataclass
class Trace:
    """
    A browsable trace.

    [Raw traces][delphyne.core.traces.Trace] contain all the information
    necessary to recompute a trace but are not easily manipulated by
    tools. In comparison, [these][delphyne.analysis.feedback.Trace]
    offer a more redundant but also more explicit view. This module
    provides a way to convert a trace from the former format to the
    latter.

    Attributes:
        nodes: A mapping from node ids to their description.

    !!! info
        A browsable trace features answer identifiers, for which a
        meaning must be provided externally. For example, the
        demonstration interpreter also produces a mapping from answer
        ids to their position in the demonstration file. In addition,
        commands like `run_strategy` return a raw trace
        (`core.traces.Trace`) in addition to the browsable version,
        which maps answer ids to their actual content.
    """

    nodes: dict[TraceNodeId, Node]


#####
##### Demo Feedback
#####


type DemoQueryId = int
"""Index of the query in the _queries_ section of a demo."""


type DemoAnswerId = tuple[int, int]
"""A (query_id, answer_index) pair that identifies an answer in a demo."""


@dataclass
class TestFeedback:
    """
    Feedback returned by the demo interpreter for a single test.

    The test is considered successful if no diagnostic is a warning or an
    error. Most of the time, and even when unsuccessful, a test stops at
    a given node, which can be inspected in the UI and which is
    indicated in field `node_id`.

    Attributes:
        diagnostics: List of diagnostics for the test.
        node_id: Identifier of the node where the test stopped.
    """

    diagnostics: list[Diagnostic]
    node_id: TraceNodeId | None


@dataclass
class ImplicitAnswer:
    """
    An implicit answer that is not part of the demonstration but was
    generated on the fly.

    The VSCode extension then offers to add such answers explicitly in
    the demonstration. This is particularly useful for handling
    `Compute` nodes in demonstrations.

    Attributes:
        query_name: Query name.
        query_args: Arguments passed to the query.
        answer: The implicit answer value, as a raw string (mode `None`
            is implicitly used for parsing).
    """

    # TODO: generalize implicit answers to accept full answers instead
    # of just a string? In particular, other modes could be used.

    query_name: str
    query_args: dict[str, object]
    answer: str


type ImplicitAnswerCategory = Literal["computations", "fetched"] | str
"""
Category of implicit answers.

The UI allows adding all answers within a given category to the
demonstration. The `computations` category is used for answers of
computation nodes (see `Compute`), while the `fetched` category is
used for answers that were fetched from external example sources (see
`using` attribute for demonstrations).
"""


@dataclass(kw_only=True)
class StrategyDemoFeedback:
    """
    Feedback sent by the server for each strategy demonstration.

    Attributes:
        kind: Always "strategy".
        trace: The resulting browsable trace, which includes all visited
            nodes.
        answer_refs: A mapping from answer ids featured in the
            trace to the position of the corresponding answer in the
            demonstration. This mapping may be **partial**. For example,
            using value hints (e.g., `#flag_value`) forces the
            demonstration interpreter to create answers on the fly that
            are not part of the demonstration.
        saved_nodes: Nodes saved using the `save` test instruction.
        test_feedback: Feedback for each test in the demonstration.
        global_diagnostics: Diagnostics that apply to the whole
            demonstration (individual tests have their own diagnostics).
        query_diagnostics: Diagnostics attached to specific queries.
        answer_diagnostics: Diagnostics attached to specific answers.
        implicit_answers: Implicit answers that were generated on the fly
            and that can be explicitly added to the demonstration,
            grouped by category. The dictionary should have no empty
            value: each mentioned catefory should have at least one
            implicit answer.
    """

    kind: Literal["strategy"]
    trace: Trace
    answer_refs: dict[TraceAnswerId, DemoAnswerId]
    saved_nodes: dict[str, TraceNodeId]
    test_feedback: list[TestFeedback]
    global_diagnostics: list[Diagnostic]
    query_diagnostics: list[tuple[DemoQueryId, Diagnostic]]
    answer_diagnostics: list[tuple[DemoAnswerId, Diagnostic]]
    implicit_answers: dict[ImplicitAnswerCategory, list[ImplicitAnswer]]


@dataclass(kw_only=True)
class QueryDemoFeedback:
    """
    Feedback sent by the server for a standalone query demonstration.

    Attributes:
        kind: Always "query".
        diagnostics: Global diagnostics.
        answer_diagnostics: Diagnostics attached to specific answers.
    """

    kind: Literal["query"]
    diagnostics: list[Diagnostic]
    answer_diagnostics: list[tuple[int, Diagnostic]]


type DemoFeedback = StrategyDemoFeedback | QueryDemoFeedback
"""
Feedback sent by the server for each demonstration in a file.
"""
