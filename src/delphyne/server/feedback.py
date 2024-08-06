"""
Type definitions for the feedback returned by the Delphyne server.
"""

from dataclasses import dataclass
from typing import Literal


#####
##### Diagnostic types
#####


type DiagnosticType = Literal["error", "warning", "info"]


type Diagnostic = tuple[DiagnosticType, str]


#####
##### Browsable trace
#####


type TraceNodeId = int
"""Global node id, as set by `core.tracing.Tracer`."""


type TraceAnswerId = int
"""Global answer id, as set by `core.tracing.Tracer`."""


type TraceActionId = int
"""Index of an action within a given node."""


type TraceNodePropertyId = int
"""Index of a property within a given node."""


@dataclass
class ValueRepr:
    """
    We allow providing several representations for Python objects:
    short, one-liner string descriptions, detailed descriptions, JSON
    representation...
    """

    short: str
    long: str | None
    json_provided: bool
    json: object


@dataclass
class Reference:
    """
    A reference to a choice or to a value.
    """

    with_ids: str
    with_hints: str | None


@dataclass
class Data:
    """Generic property that displays some data."""

    kind: Literal["data"]
    content: str


@dataclass
class Subtree:
    """
    A sub-tree (either opaque or embedded).
    """

    kind: Literal["subtree"]
    strategy: str
    args: dict[str, ValueRepr]
    node_id: TraceNodeId | None  # None if the subtree hasn't been explored


@dataclass
class Answer:
    id: TraceAnswerId
    hint: tuple[()] | tuple[str] | None
    value: ValueRepr


@dataclass
class Query:
    """
    Information about a query.
    """

    kind: Literal["query"]
    name: str
    args: dict[str, object]
    answers: list[Answer]


type NodeProperty = Data | Subtree | Query
"""
TODO: add FiniteChoice
"""


@dataclass
class Action:
    """
    Notes:
      - A list of hints is typically used as a label.
      - Storing related nodes is important to populate the path view.
      - Storing related answers is useful to detect useless answers and
        to implement a "Jump to node" action on answers.
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
    | tuple[Literal["sub"], TraceNodeId, TraceNodePropertyId]
)


@dataclass
class Node:
    """
    Notes:
    - Node labels can be used as selectors in test commands.
    - Node properties are labelled by the associated `ValueRef`.
    """

    kind: str
    success_value: ValueRepr | None
    summary_message: str | None
    leaf_node: bool
    label: str | None
    properties: list[tuple[Reference, NodeProperty]]
    actions: list[Action]
    origin: NodeOrigin


@dataclass
class Trace:
    """
    A browsable trace.

    Note: alongside with a trace, the server may want to send a separate
    mapping from answer IDs to demo references or oracle answers.
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
    The test is considered successful if no diagnostic is a warning or
    error. Most of the time and even when unsuccessful, a test stops at
    a given node, which is indicated in `node_id`.
    """

    diagnostics: list[Diagnostic]
    node_id: TraceNodeId | None


@dataclass
class DemoFeedback:
    """
    Feedback sent by the server for each demonstration.
    """

    trace: Trace
    answer_refs: dict[TraceAnswerId, DemoAnswerId]
    saved_nodes: dict[str, TraceNodeId]
    test_feedback: list[TestFeedback]
    global_diagnostics: list[Diagnostic]
    query_diagnostics: list[tuple[DemoQueryId, Diagnostic]]
    answer_diagnostics: list[tuple[DemoAnswerId, Diagnostic]]
