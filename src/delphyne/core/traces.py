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
from datetime import datetime
from typing import Any, Literal, TypeGuard

from delphyne.core import irefs, parse, refs
from delphyne.core.trees import AttachedQuery, Tree
from delphyne.utils.typing import pydantic_dump


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

    node: irefs.NodeId
    space: irefs.SpaceRef | None


#####
##### Exportable traces
#####


type NodeOriginStr = str
"""
A concise, serialized representation for `NodeOrigin`.

Can be parsed back using `parse.node_origin`.
"""


@dataclass(kw_only=True)
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
        query: The query name, if available.
        args: The query arguments, if available.
    """

    node: int
    space: str
    answers: dict[int, refs.Answer]
    query: str | None = None
    args: dict[str, Any] | None = None


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

    node: irefs.NodeId
    ref: irefs.SpaceRef


type _SerializedQuery = tuple[str, dict[str, Any]]
"""
A serialized representation of a query, as a pair of a query name and of
JSON-serialized arguments.
"""


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

    !!! note
        `answer_ids` can be nonempty while `answers` is empty, since
        one must be able to include unanswered queries in the trace.
    """

    GLOBAL_ORIGIN_ID = irefs.NodeId(0)

    def __init__(self):
        """
        Create an empty trace.
        """
        self.nodes: dict[irefs.NodeId, irefs.NodeOrigin] = {}
        self.node_ids: dict[irefs.NodeOrigin, irefs.NodeId] = {}
        self.answers: dict[
            irefs.AnswerId, tuple[QueryOrigin, refs.Answer]
        ] = {}
        self.answer_ids: dict[
            QueryOrigin, dict[refs.Answer, irefs.AnswerId]
        ] = {}
        self.serialized_queries: dict[QueryOrigin, _SerializedQuery] = {}
        self._last_node_id: int = 0
        self._last_answer_id: int = 0

    @staticmethod
    def load(trace: ExportableTrace) -> "Trace":
        """
        Load a trace from an exportable representation.
        """
        ret = Trace()
        for id, origin_str in trace.nodes.items():
            origin = parse.node_origin(origin_str)
            node_id = irefs.NodeId(id)
            ret.nodes[node_id] = origin
            ret.node_ids[origin] = node_id
        for q in trace.queries:
            origin = QueryOrigin(
                irefs.NodeId(q.node),
                parse.id_based_space_ref(q.space),
            )
            ret.answer_ids[origin] = {}
            for ans_id, ans in q.answers.items():
                ret.answers[irefs.AnswerId(ans_id)] = (origin, ans)
                ret.answer_ids[origin][ans] = irefs.AnswerId(ans_id)
            if q.query is not None and q.args is not None:
                serialized = (q.query, q.args)
                ret.serialized_queries[origin] = serialized
        ret._last_node_id = max((id.id for id in ret.nodes), default=0)
        ret._last_answer_id = max((id.id for id in ret.answers), default=0)
        return ret

    def fresh_or_cached_node_id(
        self, origin: irefs.NodeOrigin
    ) -> irefs.NodeId:
        """
        Obtain the identifier of a node described by its origin.
        Create a new identifier on the fly if it does not exist yet.
        """
        if origin in self.node_ids:
            return self.node_ids[origin]
        else:
            self._last_node_id += 1
            id = irefs.NodeId(self._last_node_id)
            self.nodes[id] = origin
            self.node_ids[origin] = id
            return id

    def fresh_or_cached_answer_id(
        self, answer: refs.Answer, origin: QueryOrigin
    ) -> irefs.AnswerId:
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
            id = irefs.AnswerId(self._last_answer_id)
            self.answers[id] = (origin, answer)
            self.answer_ids[origin][answer] = id
            return id

    def register_query(
        self, ref: refs.GlobalSpacePath, query: _SerializedQuery
    ) -> None:
        """
        Ensure that a query appears in the trace, even if not answers
        are associated with it yet. Optionally, attach a serialized
        query representation to the trace.

        This is particularly useful for the demonstration interpreter.
        Indeed, when a test gets stuck on an unanswered query, it is
        desirable for this query to be part of the returned trace so
        that the user can visualize it.
        """
        origin = self._convert_query_origin(ref)
        if origin not in self.answer_ids:
            self.answer_ids[origin] = {}
        self.serialized_queries[origin] = query

    def export(self, add_serialized_queries: bool = True) -> ExportableTrace:
        """
        Export a trace into a lightweight, serializable format.
        """
        nodes = {id.id: str(origin) for id, origin in self.nodes.items()}
        queries: list[ExportableQueryInfo] = []
        for q, a in self.answer_ids.items():
            if add_serialized_queries:
                serialized = self.serialized_queries.get(q, (None, None))
            else:
                serialized = (None, None)
            ref = str(q.ref)
            answers = {id.id: value for value, id in a.items()}
            queries.append(
                ExportableQueryInfo(
                    node=q.node.id,
                    space=ref,
                    answers=answers,
                    query=serialized[0],
                    args=serialized[1],
                )
            )
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
            id_bis = self.convert_global_node_path(expanded)
            assert id == id_bis

    def check_roundabout_consistency(self) -> None:
        """
        Perform a sanity check, before and after serializing and
        desarializing it.
        """
        self.check_consistency()
        exportable = self.export()
        copy = Trace.load(exportable)
        copy.check_consistency()
        exportable_copy = copy.export()
        if exportable != exportable_copy:
            print("Original exportable trace:")
            print(exportable)
            print("Exportable trace after round-trip:")
            print(exportable_copy)
            assert False

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

    def _convert_query_origin(self, ref: refs.GlobalSpacePath) -> QueryOrigin:
        """
        Convert a full, global space reference denoting a quey origin
        into an id-based reference.
        """
        id = self.convert_global_node_path(ref[0])
        space = self._convert_space_ref(id, ref[1])
        origin = QueryOrigin(id, space)
        return origin

    def convert_answer_ref(
        self, ref: tuple[refs.GlobalSpacePath, refs.Answer]
    ) -> irefs.AnswerId:
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
    ) -> irefs.NodeId:
        """
        Convert a full, global node reference into an id-based one.
        """
        id = Trace.GLOBAL_ORIGIN_ID
        for space, node_path in path:
            space_ref = self._convert_space_ref(id, space)
            id = self.fresh_or_cached_node_id(
                irefs.NestedTreeOf(id, space_ref)
            )
            id = self._convert_node_path(id, node_path)
        return id

    def convert_global_space_path(
        self, path: refs.GlobalSpacePath
    ) -> irefs.SpaceRef:
        """
        Convert a full global space reference into an id-based one.
        """
        node_path, space_ref = path
        id = self.convert_global_node_path(node_path)
        return self._convert_space_ref(id, space_ref)

    def _convert_node_path(
        self, id: irefs.NodeId, path: refs.NodePath
    ) -> irefs.NodeId:
        """
        Convert a full local node path into an identifier, relative to a
        given node.
        """
        for a in path:
            action_ref = self._convert_value_ref(id, a)
            id = self.fresh_or_cached_node_id(irefs.ChildOf(id, action_ref))
        return id

    def _convert_space_ref(
        self, id: irefs.NodeId, ref: refs.SpaceRef
    ) -> irefs.SpaceRef:
        """
        Convert a full local space reference into an id-based one, relative
        to a given node.
        """
        args = tuple(self._convert_value_ref(id, a) for a in ref.args)
        return irefs.SpaceRef(ref.name, args)

    def _convert_atomic_value_ref(
        self, id: irefs.NodeId, ref: refs.AtomicValueRef
    ) -> irefs.AtomicValueRef:
        """
        Convert a full local atomic value reference into an id-based one,
        relative to a given node.
        """
        if isinstance(ref, refs.IndexedRef):
            return irefs.IndexedRef(
                self._convert_atomic_value_ref(id, ref.ref), ref.index
            )
        else:
            return self._convert_space_element_ref(id, ref)

    def _convert_value_ref(
        self, id: irefs.NodeId, ref: refs.ValueRef
    ) -> irefs.ValueRef:
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
        self, id: irefs.NodeId, ref: refs.SpaceElementRef
    ) -> irefs.SpaceElementRef:
        """
        Convert a full local space element reference into an id-based one,
        relative to a given node.
        """
        space = self._convert_space_ref(id, ref.space)
        match ref.element:
            case refs.Answer():
                assert space is not None
                origin = QueryOrigin(id, space)
                element = self.fresh_or_cached_answer_id(ref.element, origin)
            case tuple():
                assert space is not None
                nested_root_orig = irefs.NestedTreeOf(id, space)
                nested_root = self.fresh_or_cached_node_id(nested_root_orig)
                element = self._convert_node_path(nested_root, ref.element)
        return irefs.SpaceElementRef(space, element)

    ### Reverse direction: expanding id-based references into full ones.

    def expand_space_ref(
        self, id: irefs.NodeId, ref: irefs.SpaceRef
    ) -> refs.SpaceRef:
        """
        Convert a local id-based space reference into a full one,
        relative to a given node.
        """
        args = tuple(self.expand_value_ref(id, a) for a in ref.args)
        return refs.SpaceRef(ref.name, args)

    def expand_value_ref(
        self, id: irefs.NodeId, ref: irefs.ValueRef
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

    def expand_node_id(self, id: irefs.NodeId) -> refs.GlobalNodePath:
        """
        Convert a node identifier into a full, global node reference.
        """
        rev_path: list[tuple[refs.SpaceRef, refs.NodePath]] = []
        while id != Trace.GLOBAL_ORIGIN_ID:
            id, space, path = self._recover_path(id)
            rev_path.append((space, path))
        return tuple(reversed(rev_path))

    def _expand_atomic_value_ref(
        self, id: irefs.NodeId, ref: irefs.AtomicValueRef
    ) -> refs.AtomicValueRef:
        """
        Convert a local id-based atomic value reference into a full one,
        relative to a given node.
        """
        if isinstance(ref, irefs.IndexedRef):
            return refs.IndexedRef(
                self._expand_atomic_value_ref(id, ref.ref), ref.index
            )
        else:
            return self._expand_space_element_ref(id, ref)

    def _expand_space_element_ref(
        self, id: irefs.NodeId, ref: irefs.SpaceElementRef
    ) -> refs.SpaceElementRef:
        """
        Convert a local id-based space element reference into a full
        one, relative to a given node.
        """
        assert isinstance(ref, irefs.SpaceElementRef)
        assert ref.space is not None
        space = self.expand_space_ref(id, ref.space)
        match ref.element:
            case irefs.AnswerId():
                _orig, ans = self.answers[ref.element]
                element = ans
            case irefs.NodeId():
                orig, _, element = self._recover_path(ref.element)
                assert orig == id
        return refs.SpaceElementRef(space, element)

    def _recover_path(
        self, dst: irefs.NodeId
    ) -> tuple[irefs.NodeId, refs.SpaceRef, refs.NodePath]:
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
                case irefs.ChildOf(before, action):
                    rev_path.append(self.expand_value_ref(before, action))
                    dst = before
                case irefs.NestedTreeOf(orig, space):
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

    children: dict[irefs.NodeId, dict[irefs.ValueRef, irefs.NodeId]] = field(
        default_factory=lambda: defaultdict(lambda: {})
    )
    nested_trees: dict[irefs.NodeId, dict[irefs.SpaceRef, irefs.NodeId]] = (
        field(default_factory=lambda: defaultdict(lambda: {}))
    )

    @staticmethod
    def make(trace: Trace) -> "TraceReverseMap":
        """
        Build a reverse map from a trace.
        """
        map = TraceReverseMap()
        for child_id, origin in trace.nodes.items():
            match origin:
                case irefs.ChildOf(parent_id, action):
                    map.children[parent_id][action] = child_id
                case irefs.NestedTreeOf(parent_id, space):
                    map.nested_trees[parent_id][space] = child_id
        return map


#####
##### Tracer
#####


type LogLevel = Literal["trace", "debug", "info", "warn", "error"]


def log_level_greater_or_equal(lhs: LogLevel, rhs: LogLevel) -> bool:
    levels = ["trace", "debug", "info", "warn", "error"]
    return levels.index(lhs) >= levels.index(rhs)


def valid_log_level(level: str) -> TypeGuard[LogLevel]:
    if level in ["trace", "debug", "info", "warn", "error"]:
        return True
    return False


@dataclass(frozen=True, kw_only=True)
class LogMessage:
    """
    A log message.

    Attributes:
        message: The message to log.
        time: Time at which the message was produced
        metadata: Optional metadata associated with the message, as an
            object that can be serialized to JSON using Pydantic.
        location: An optional location in the strategy tree where the
            message was logged, if applicable.
    """

    message: str
    level: LogLevel
    time: datetime
    metadata: object | None = None
    location: ShortLocation | None = None


@dataclass(frozen=True, kw_only=True)
class ExportableLogMessage:
    """
    An exportable log message, as a dataclass whose fields are JSON
    values (as opposed to `LogMessage`) and is thus easier to export.
    """

    message: str
    level: LogLevel
    time: datetime | None = None
    node: int | None = None
    space: str | None = None
    metadata: object | None = None  # JSON value


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
            messages in between.
    """

    # TODO: there are cleaner ways to achieve good message order beyong
    # exposing the lock.

    def __init__(self, log_level: LogLevel = "info"):
        """
        Parameters:
            log_level: The minimum severity level of messages to log.
        """
        self.trace = Trace()
        self.messages: list[LogMessage] = []
        self.log_level: LogLevel = log_level

        # Different threads may be logging information or appending to
        # the trace in parallel.
        self.lock = threading.RLock()

    def global_node_id(self, node: refs.GlobalNodePath) -> irefs.NodeId:
        """
        Ensure that a node at a given reference is present in the trace
        and return the corresponding node identififier.
        """
        with self.lock:
            return self.trace.convert_global_node_path(node)

    def trace_node(self, node: refs.GlobalNodePath) -> None:
        """
        Ensure that a node at a given reference is present in the trace.

        Returns the associated node identifier.

        See `tracer_hook` for registering a hook that automatically
        calls this method on all encountered nodes.
        """
        self.global_node_id(node)

    def trace_query(self, query: AttachedQuery[Any]) -> None:
        """
        Ensure that a query at a given reference is present in the
        trace, even if no answer is provided for it.
        """
        serialized = (query.query.query_name(), query.query.serialize_args())
        with self.lock:
            self.trace.register_query(query.ref, serialized)

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
        level: LogLevel,
        message: str,
        metadata: object | None = None,
        location: Location | None = None,
    ):
        """
        Log a message, with optional metadata and location information.
        The metadata must be exportable to JSON using Pydantic.
        """
        if not log_level_greater_or_equal(level, self.log_level):
            return
        time = datetime.now()
        with self.lock:
            short_location = None
            if location is not None:
                short_location = self.trace.convert_location(location)
            self.messages.append(
                LogMessage(
                    message=message,
                    level=level,
                    time=time,
                    metadata=metadata,
                    location=short_location,
                )
            )

    def export_log(
        self, *, remove_timing_info: bool = False
    ) -> Iterable[ExportableLogMessage]:
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
                        space = str(loc.space)
                yield ExportableLogMessage(
                    message=m.message,
                    level=m.level,
                    time=m.time if not remove_timing_info else None,
                    node=node,
                    space=space,
                    metadata=pydantic_dump(object, m.metadata),
                )

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
