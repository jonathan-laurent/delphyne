"""
Representing, exporting and importing traces.

A trace (`Trace`) denotes a collection of reachable nodes and spaces,
which is encoded in a concise way by introducing unique identifiers for
answers and nodes.
"""

import threading
from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from delphyne.core import pprint, refs
from delphyne.core.trees import Tree


@dataclass(frozen=True)
class Location:
    """
    A **full**, global reference to either a node or a space.

    This is useful in particular for attaching location information to
    logging messages.
    """

    node: refs.GlobalNodePath
    space: refs.SpaceRef | None


@dataclass(frozen=True)
class ShortLocation:
    """
    An **id-based**, global reference to either a node or a space.

    This is the id-based counterpart of `Location`. Policies typically
    log messages with `Location` values attached (since trees feature
    full references), which are then converted into `ShortLocation` in
    the final exportable log.
    """

    node: refs.NodeId
    space: refs.SpaceRef | None


#####
##### Exportable traces
#####


type NodeOriginStr = str
"""
A concise, serialized representation for `NodeOrigin`.
"""


@dataclass
class ExportableQueryInfo:
    """
    Information about a query encountered in an exportable trace.

    Attributes:
        node: Identifier of the node that the query is attached to.
        space: Local, id-based reference of the space that the query
            belongs to. Serialized using `pprint.space_ref`.
        answers: Mapping from answer identifiers to actual answers.
            Answer identifiers are unique across a whole exportable
            trace (and not only across an `ExportableQueryInfo` value).
    """

    node: int
    space: str
    answers: dict[int, refs.Answer]


@dataclass
class ExportableTrace:
    """
    A lightweight trace format that can be easily exported to JSON/YAML.

    Attributes:
        nodes: a mapping from node ids to serialized origin information.
        queries: a list of encountered queries with associated answers.
    """

    nodes: dict[int, NodeOriginStr]
    queries: list[ExportableQueryInfo]


#####
##### Traces
#####


@dataclass(frozen=True)
class QueryOrigin:
    """
    A global, id-based reference to the space induced by a query.
    """

    node: refs.NodeId
    ref: refs.SpaceRef


class Trace:
    """
    A collection of reachable nodes and spaces, which is encoded in a
    concise way by introducing unique identifiers for answers and nodes.

    Traces are mutable. Methods are provided to convert full references
    into id-based references, creating fresh identifiers for new nodes
    and queries on the fly. Backward conversion methods are also
    provided for converting id-based references back into full
    references (assuming id-based references are valid, without which
    these methods fail with assertion errors).

    Attributes:
        nodes: a mapping from node identifiers to their origin.
        node_ids: reverse map of `nodes`.
        answers: a mapping from answer identifiers to actual answers,
            along with origin information on the associated query.
        answer_ids: reverse map of `answers`.
    """

    GLOBAL_ORIGIN_ID = refs.NodeId(0)

    def __init__(self):
        """
        Create an empty trace.
        """
        self.nodes: dict[refs.NodeId, refs.NodeOrigin] = {}
        self.node_ids: dict[refs.NodeOrigin, refs.NodeId] = {}
        self.answers: dict[refs.AnswerId, tuple[QueryOrigin, refs.Answer]] = {}
        self.answer_ids: dict[
            QueryOrigin, dict[refs.Answer, refs.AnswerId]
        ] = {}
        self._last_node_id: int = 0
        self._last_answer_id: int = 0

    def fresh_or_cached_node_id(self, origin: refs.NodeOrigin) -> refs.NodeId:
        """
        Obtain the identifier of a node described by its origin.
        Create a new identifier on the fly if it does not exist yet.
        """
        if origin in self.node_ids:
            return self.node_ids[origin]
        else:
            self._last_node_id += 1
            id = refs.NodeId(self._last_node_id)
            self.nodes[id] = origin
            self.node_ids[origin] = id
            return id

    def fresh_or_cached_answer_id(
        self, answer: refs.Answer, origin: QueryOrigin
    ) -> refs.AnswerId:
        """
        Obtain the identifier of an answer, given its content and the
        origin of the query that it corresponds to. Create a new, fresh
        identifier on the fly if it does not exist yet.
        """
        if origin not in self.answer_ids:
            self.answer_ids[origin] = {}
        if answer in self.answer_ids[origin]:
            return self.answer_ids[origin][answer]
        else:
            self._last_answer_id += 1
            id = refs.AnswerId(self._last_answer_id)
            self.answers[id] = (origin, answer)
            self.answer_ids[origin][answer] = id
            return id

    def register_query(self, origin: QueryOrigin) -> None:
        """
        Ensure that a query appears in the trace, even if not answers
        are associated with it yet.

        This is particularly useful for the demonstration interpreter.
        Indeed, when a test gets stuck on an unanswered query, it is
        desirable for this query to be part of the returned trace so
        that the user can visualize it.
        """
        if origin not in self.answer_ids:
            self.answer_ids[origin] = {}

    def export(self) -> ExportableTrace:
        """
        Export a trace into a lightweight, serializable format.
        """
        nodes = {
            id.id: pprint.node_origin(origin)
            for id, origin in self.nodes.items()
        }
        queries: list[ExportableQueryInfo] = []
        for q, a in self.answer_ids.items():
            ref = pprint.space_ref(q.ref)
            answers = {id.id: value for value, id in a.items()}
            queries.append(ExportableQueryInfo(q.node.id, ref, answers))
        return ExportableTrace(nodes, queries)

    def check_consistency(self) -> None:
        """
        Perform a sanity check on the trace.

        Each node identifier is expanded into a full reference and then
        converted back to an identifier, which must be equal to the
        original one.
        """
        for id in self.nodes:
            expanded = self.expand_node_id(id)
            assert id == self.convert_global_node_path(expanded)

    ### Convert full references into id-based references

    def convert_location(self, location: Location) -> ShortLocation:
        """
        Convert a full location into an id-based one.
        """
        id = self.convert_global_node_path(location.node)
        space = None
        if location.space is not None:
            space = self._convert_space_ref(id, location.space)
        return ShortLocation(id, space)

    def convert_query_origin(self, ref: refs.GlobalSpacePath) -> QueryOrigin:
        """
        Convert a full, global space reference denoting a quey origin
        into an id-based reference.
        """
        id = self.convert_global_node_path(ref[0])
        space = self._convert_space_ref(id, ref[1])
        origin = QueryOrigin(id, space)
        self.register_query(origin)
        return origin

    def convert_answer_ref(
        self, ref: tuple[refs.GlobalSpacePath, refs.Answer]
    ) -> refs.AnswerId:
        """
        Convert a full answer reference into an answer id.
        """
        node_path, space = ref[0]
        id = self.convert_global_node_path(node_path)
        space = self._convert_space_ref(id, space)
        origin = QueryOrigin(id, space)
        return self.fresh_or_cached_answer_id(ref[1], origin)

    def convert_global_node_path(
        self, path: refs.GlobalNodePath
    ) -> refs.NodeId:
        """
        Convert a full, global node reference into an id-based one.
        """
        id = Trace.GLOBAL_ORIGIN_ID
        for space, node_path in path:
            space_ref = self._convert_space_ref(id, space)
            id = self.fresh_or_cached_node_id(refs.NestedTreeOf(id, space_ref))
            id = self._convert_node_path(id, node_path)
        return id

    def convert_global_space_path(
        self, path: refs.GlobalSpacePath
    ) -> refs.SpaceRef:
        """
        Convert a full global space reference into an id-based one.
        """
        node_path, space_ref = path
        id = self.convert_global_node_path(node_path)
        return self._convert_space_ref(id, space_ref)

    def _convert_node_path(
        self, id: refs.NodeId, path: refs.NodePath
    ) -> refs.NodeId:
        """
        Convert a full local node path into an identifier, relative to a
        given node.
        """
        for a in path:
            action_ref = self._convert_value_ref(id, a)
            id = self.fresh_or_cached_node_id(refs.ChildOf(id, action_ref))
        return id

    def _convert_space_ref(
        self, id: refs.NodeId, ref: refs.SpaceRef
    ) -> refs.SpaceRef:
        """
        Convert a full local space reference into an id-based one, relative
        to a given node.
        """
        args = tuple(self._convert_value_ref(id, a) for a in ref.args)
        return refs.SpaceRef(ref.name, args)

    def _convert_atomic_value_ref(
        self, id: refs.NodeId, ref: refs.AtomicValueRef
    ) -> refs.AtomicValueRef:
        """
        Convert a full local atomic value reference into an id-based one,
        relative to a given node.
        """
        if isinstance(ref, refs.IndexedRef):
            return refs.IndexedRef(
                self._convert_atomic_value_ref(id, ref.ref), ref.index
            )
        else:
            return self._convert_space_element_ref(id, ref)

    def _convert_value_ref(
        self, id: refs.NodeId, ref: refs.ValueRef
    ) -> refs.ValueRef:
        """
        Convert a full local value reference into an id-based one,
        relative to a given node.
        """
        if ref is None:
            return None
        elif isinstance(ref, tuple):
            return tuple(self._convert_value_ref(id, a) for a in ref)
        else:
            return self._convert_atomic_value_ref(id, ref)

    def _convert_space_element_ref(
        self, id: refs.NodeId, ref: refs.SpaceElementRef
    ) -> refs.SpaceElementRef:
        """
        Convert a full local space element reference into an id-based one,
        relative to a given node.
        """
        space = None
        if ref.space is not None:
            space = self._convert_space_ref(id, ref.space)
        match ref.element:
            case refs.Answer():
                assert space is not None
                origin = QueryOrigin(id, space)
                element = self.fresh_or_cached_answer_id(ref.element, origin)
            case refs.AnswerId() | refs.NodeId():
                element = ref.element
            case refs.HintsRef():
                assert False
            case tuple():
                assert space is not None
                nested_root_orig = refs.NestedTreeOf(id, space)
                nested_root = self.fresh_or_cached_node_id(nested_root_orig)
                element = self._convert_node_path(nested_root, ref.element)
        return refs.SpaceElementRef(space, element)

    ### Reverse direction: expanding id-based references into full ones.

    def expand_space_ref(
        self, id: refs.NodeId, ref: refs.SpaceRef
    ) -> refs.SpaceRef:
        """
        Convert a local id-based space reference into a full one,
        relative to a given node.
        """
        args = tuple(self.expand_value_ref(id, a) for a in ref.args)
        return refs.SpaceRef(ref.name, args)

    def expand_value_ref(
        self, id: refs.NodeId, ref: refs.ValueRef
    ) -> refs.ValueRef:
        """
        Convert a local id-based value reference into a full one,
        relative to a given node.
        """
        if ref is None:
            return None
        elif isinstance(ref, tuple):
            return tuple(self.expand_value_ref(id, a) for a in ref)
        else:
            return self._expand_atomic_value_ref(id, ref)

    def expand_node_id(self, id: refs.NodeId) -> refs.GlobalNodePath:
        """
        Convert a node identifier into a full, global node reference.
        """
        rev_path: list[tuple[refs.SpaceRef, refs.NodePath]] = []
        while id != Trace.GLOBAL_ORIGIN_ID:
            id, space, path = self._recover_path(id)
            rev_path.append((space, path))
        return tuple(reversed(rev_path))

    def _expand_atomic_value_ref(
        self, id: refs.NodeId, ref: refs.AtomicValueRef
    ) -> refs.AtomicValueRef:
        """
        Convert a local id-based atomic value reference into a full one,
        relative to a given node.
        """
        if isinstance(ref, refs.IndexedRef):
            return refs.IndexedRef(
                self._expand_atomic_value_ref(id, ref.ref), ref.index
            )
        else:
            return self._expand_space_element_ref(id, ref)

    def _expand_space_element_ref(
        self, id: refs.NodeId, ref: refs.SpaceElementRef
    ) -> refs.SpaceElementRef:
        """
        Convert a local id-based space element reference into a full
        one, relative to a given node.
        """
        assert isinstance(ref, refs.SpaceElementRef)
        assert ref.space is not None
        space = self.expand_space_ref(id, ref.space)
        match ref.element:
            case refs.AnswerId():
                _orig, ans = self.answers[ref.element]
                element = ans
            case refs.NodeId():
                orig, _, element = self._recover_path(ref.element)
                assert orig == id
            case _:
                assert False
        return refs.SpaceElementRef(space, element)

    def _recover_path(
        self, dst: refs.NodeId
    ) -> tuple[refs.NodeId, refs.SpaceRef, refs.NodePath]:
        """
        Find the node from which the tree containing `dst` originates.

        Return the node in which the full surrounding tree is nested,
        the associated space reference, and a path to `dst` from the
        root of the surrounding tree.
        """
        rev_path: list[refs.ValueRef] = []
        while True:
            dst_origin = self.nodes[dst]
            match dst_origin:
                case refs.ChildOf(before, action):
                    rev_path.append(self.expand_value_ref(before, action))
                    dst = before
                case refs.NestedTreeOf(orig, space):
                    space = self.expand_space_ref(orig, space)
                    return orig, space, tuple(reversed(rev_path))


#####
##### Reverse Map
#####


@dataclass
class TraceReverseMap:
    """
    A mapping from node identifiers to children and nested trees.

    Attributes:
        children: maps a node identifier to a dictionary with one entry
            per child, which maps the id-based value reference of the
            associated action to the id of the subtree's root.
        nested_trees: maps a node identifier to a dictionary with one
            entry per nested tree, which maps the id-based spce
            reference of the induced space to the nested tree id.
    """

    children: dict[refs.NodeId, dict[refs.ValueRef, refs.NodeId]] = field(
        default_factory=lambda: defaultdict(lambda: {}))  # fmt: skip
    nested_trees: dict[refs.NodeId, dict[refs.SpaceRef, refs.NodeId]] = field(
        default_factory=lambda: defaultdict(lambda: {}))  # fmt: skip

    @staticmethod
    def make(trace: Trace) -> "TraceReverseMap":
        """
        Build a reverse map from a trace.
        """
        map = TraceReverseMap()
        for child_id, origin in trace.nodes.items():
            match origin:
                case refs.ChildOf(parent_id, action):
                    map.children[parent_id][action] = child_id
                case refs.NestedTreeOf(parent_id, space):
                    map.nested_trees[parent_id][space] = child_id
        return map


#####
##### Tracer
#####


@dataclass(frozen=True)
class LogMessage:
    """
    A log message.

    Attributes:
        message: The message to log.
        metadata: Optional metadata associated with the message, as a
            dictionary mapping string keys to JSON values.
        location: An optional location in the strategy tree where the
            message was logged, if applicable.
    """

    message: str
    metadata: dict[str, Any] | None = None
    location: ShortLocation | None = None


@dataclass(frozen=True)
class ExportableLogMessage:
    """
    An exportable log message, as a dataclass whose fields are JSON
    values (as opposed to `LogMessage`) and is thus easier to export.
    """

    message: str
    node: int | None
    space: str | None
    metadata: dict[str, Any] | None = None


class Tracer:
    """
    A mutable trace along with a mutable list of log messages.

    Both components are protected by a lock to ensure thread-safety
    (some policies spawn multiple concurrent threads).

    Attributes:
        trace: A mutable trace.
        messages: A mutable list of log messages.
        lock: A reentrant lock protecting access to the trace and log.
            The lock is publicly exposed so that threads can log several
            successive messages without other threads interleaving new
            messages in between (TODO: there are cleaner ways to achieve
            this).
    """

    def __init__(self):
        self.trace = Trace()
        self.messages: list[LogMessage] = []

        # Different threads may be logging information or appending to
        # the trace in parallel.
        self.lock = threading.RLock()

    def trace_node(self, node: refs.GlobalNodePath) -> None:
        """
        Ensure that a node at a given reference is present in the trace.

        See `tracer_hook` for registering a hook that automatically
        calls this method on all encountered nodes.
        """
        with self.lock:
            self.trace.convert_location(Location(node, None))

    def trace_query(self, ref: refs.GlobalSpacePath) -> None:
        """
        Ensure that a query at a given reference is present in the
        trace, even if no answer is provided for it.
        """
        with self.lock:
            self.trace.convert_query_origin(ref)

    def trace_answer(
        self, space: refs.GlobalSpacePath, answer: refs.Answer
    ) -> None:
        """
        Ensure that a given query answer is present in the trace, even
        it is is not used to reach a node.
        """
        with self.lock:
            self.trace.convert_answer_ref((space, answer))

    def log(
        self,
        message: str,
        metadata: dict[str, Any] | None = None,
        location: Location | None = None,
    ):
        """
        Log a message, with optional metadata and location information.
        The metadata must be a dictionary of JSON values.
        """
        with self.lock:
            short_location = None
            if location is not None:
                short_location = self.trace.convert_location(location)
            self.messages.append(LogMessage(message, metadata, short_location))

    def export_log(self) -> Iterable[ExportableLogMessage]:
        """
        Export the log into an easily serializable format.
        """
        with self.lock:
            for m in self.messages:
                node = None
                space = None
                if (loc := m.location) is not None:
                    node = loc.node.id
                    if loc.space is not None:
                        space = pprint.space_ref(loc.space)
                yield ExportableLogMessage(m.message, node, space, m.metadata)

    def export_trace(self) -> ExportableTrace:
        """
        Export the trace into an easily serializable format.
        """
        with self.lock:
            return self.trace.export()


def tracer_hook(tracer: Tracer) -> Callable[[Tree[Any, Any, Any]], None]:
    """
    Standard hook to be passed to `TreeMonitor` to automatically log
    visited nodes into a trace.
    """
    return lambda tree: tracer.trace_node(tree.ref)
