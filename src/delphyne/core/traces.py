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
from typing import Any, Literal, TypeGuard, assert_type

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
    space: irefs.SpaceId | None


#####
##### Exportable traces
#####


type NodeOriginStr = str
"""
A concise, serialized representation for `NodeOrigin`.

Can be parsed back using `parse.node_origin`.
"""


type SpaceOriginStr = str
"""
A concise, serialized representation for `SpaceOrigin`.

Can be parsed back using `parse.space_origin`.
"""


@dataclass(frozen=True)
class _ExportableLocatedAnswer:
    space: int
    answer: refs.Answer


@dataclass
class ExportableTrace:
    """
    A lightweight trace format that can be easily exported to JSON/YAML.

    Attributes:
        nodes: a mapping that defines node identifiers
        spaces: a mapping that defines space identifiers
        answers: a mapping that defines answer identifiers
    """

    nodes: dict[int, NodeOriginStr]
    spaces: dict[int, SpaceOriginStr]
    answers: dict[int, _ExportableLocatedAnswer]


#####
##### Traces
#####


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
        spaces: a mapping from space identifiers to actual space
            definitions
        space_ids: reverse map of `spaces`.

    !!! note
        `answer_ids` can be nonempty while `answers` is empty, since
        one must be able to include unanswered queries in the trace.
    """

    GLOBAL_ORIGIN_ID = irefs.NodeId(0)
    MAIN_SPACE_ID = irefs.SpaceId(0)
    MAIN_SPACE_ORIGIN = irefs.SpaceOrigin(GLOBAL_ORIGIN_ID, irefs.MAIN_SPACE)

    def __init__(self):
        """
        Create an empty trace.
        """
        self.nodes: dict[irefs.NodeId, irefs.NodeOrigin] = {}
        self.node_ids: dict[irefs.NodeOrigin, irefs.NodeId] = {}
        self.answers: dict[irefs.AnswerId, irefs.LocatedAnswer] = {}
        self.answer_ids: dict[irefs.LocatedAnswer, irefs.AnswerId] = {}
        self.spaces: dict[irefs.SpaceId, irefs.SpaceOrigin] = {}
        self.space_ids: dict[irefs.SpaceOrigin, irefs.SpaceId] = {}
        self._last_node_id: int = 0
        self._last_answer_id: int = 0
        self._last_space_id: int = 0

        self.spaces[Trace.MAIN_SPACE_ID] = Trace.MAIN_SPACE_ORIGIN
        self.space_ids[Trace.MAIN_SPACE_ORIGIN] = Trace.MAIN_SPACE_ID

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
        for id, origin_str in trace.spaces.items():
            origin = parse.space_origin(origin_str)
            space_id = irefs.SpaceId(id)
            ret.spaces[space_id] = origin
            ret.space_ids[origin] = space_id
        for id, located_s in trace.answers.items():
            answer_id = irefs.AnswerId(id)
            space_id = irefs.SpaceId(located_s.space)
            located = irefs.LocatedAnswer(space_id, located_s.answer)
            ret.answers[answer_id] = located
            ret.answer_ids[located] = answer_id
        ret._last_node_id = max((id.id for id in ret.nodes), default=0)
        ret._last_space_id = max((id.id for id in ret.spaces), default=0)
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

    def fresh_or_cached_space_id(
        self, origin: irefs.SpaceOrigin
    ) -> irefs.SpaceId:
        """
        Obtain the identifier of a space, given its origin. Create a new,
        fresh identifier on the fly if it does not exist yet.
        """
        if origin in self.space_ids:
            return self.space_ids[origin]
        else:
            self._last_space_id += 1
            id = irefs.SpaceId(self._last_space_id)
            self.spaces[id] = origin
            self.space_ids[origin] = id
            return id

    def fresh_or_cached_answer_id(
        self, answer: irefs.LocatedAnswer
    ) -> irefs.AnswerId:
        """
        Obtain the identifier of an answer, given its content and the
        origin of the query that it corresponds to. Create a new, fresh
        identifier on the fly if it does not exist yet.
        """
        if answer in self.answer_ids:
            return self.answer_ids[answer]
        else:
            self._last_answer_id += 1
            id = irefs.AnswerId(self._last_answer_id)
            self.answers[id] = answer
            self.answer_ids[answer] = id
            return id

    def export(self) -> ExportableTrace:
        """
        Export a trace into a lightweight, serializable format.
        """
        nodes = {id.id: str(origin) for id, origin in self.nodes.items()}
        spaces = {id.id: str(origin) for id, origin in self.spaces.items()}
        answers = {
            id.id: _ExportableLocatedAnswer(located.space.id, located.answer)
            for id, located in self.answers.items()
        }
        return ExportableTrace(nodes, spaces, answers)

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

    def _convert_space_ref(
        self, id: irefs.NodeId, ref: refs.SpaceRef
    ) -> irefs.SpaceId:
        """
        Convert a full local space reference into an id-based one, relative
        to a given node.
        """
        args = tuple(self._convert_value_ref(id, a) for a in ref.args)
        space_ref = irefs.SpaceRef(ref.name, args)
        return self.fresh_or_cached_space_id(irefs.SpaceOrigin(id, space_ref))

    def convert_global_space_path(
        self, ref: refs.GlobalSpacePath
    ) -> irefs.SpaceId:
        """
        Convert a full, global space reference denoting a quey origin
        into an id-based reference.
        """
        id = self.convert_global_node_path(ref[0])
        return self._convert_space_ref(id, ref[1])

    def convert_answer_ref(
        self, ref: tuple[refs.GlobalSpacePath, refs.Answer]
    ) -> irefs.AnswerId:
        """
        Convert a full answer reference into an answer id.
        """
        space = self.convert_global_space_path(ref[0])
        located = irefs.LocatedAnswer(space, ref[1])
        return self.fresh_or_cached_answer_id(located)

    def convert_global_node_path(
        self, path: refs.GlobalNodePath
    ) -> irefs.NodeId:
        """
        Convert a full, global node reference into an id-based one.
        """
        id = Trace.GLOBAL_ORIGIN_ID
        for space, node_path in path:
            space_id = self._convert_space_ref(id, space)
            id = self.fresh_or_cached_node_id(irefs.NestedIn(space_id))
            id = self._convert_node_path(id, node_path)
        return id

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
        space_id = self._convert_space_ref(id, ref.space)
        match ref.element:
            case refs.Answer():
                located = irefs.LocatedAnswer(space_id, ref.element)
                element = self.fresh_or_cached_answer_id(located)
            case tuple():
                assert_type(ref.element, refs.NodePath)
                nested_root_orig = irefs.NestedIn(space_id)
                nested_root = self.fresh_or_cached_node_id(nested_root_orig)
                element = self._convert_node_path(nested_root, ref.element)
        return irefs.SpaceElementRef(space_id, element)

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

    def expand_space_id(self, id: irefs.SpaceId) -> refs.SpaceRef:
        origin = self.spaces[id]
        return self.expand_space_ref(origin.node, origin.space)

    def expand_global_space_id(
        self, id: irefs.SpaceId
    ) -> refs.GlobalSpacePath:
        origin = self.spaces[id]
        node_path = self.expand_node_id(origin.node)
        space_ref = self.expand_space_ref(origin.node, origin.space)
        return (node_path, space_ref)

    def _expand_located_answer_ref(
        self, ans: irefs.LocatedAnswer
    ) -> refs.GlobalAnswerRef:
        space = self.expand_global_space_id(ans.space)
        return (space, ans.answer)

    def expand_answer_id(self, ans: irefs.AnswerId) -> refs.GlobalAnswerRef:
        located = self.answers[ans]
        return self._expand_located_answer_ref(located)

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
        space = self.expand_space_id(ref.space)
        match ref.element:
            case irefs.AnswerId():
                located = self.answers[ref.element]
                element = located.answer
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
                case irefs.NestedIn(space_id):
                    space = self.expand_space_id(space_id)
                    orig = self.spaces[space_id].node
                    return orig, space, tuple(reversed(rev_path))

    ### Extracting local space elements

    def space_elements_in_value_ref(
        self, ref: irefs.ValueRef
    ) -> Iterable[irefs.SpaceElementRef]:
        """
        Enumerate all local space elements that are used to define a
        value.

        Duplicate values can be returned.
        """

        if ref is None:
            pass
        elif isinstance(ref, tuple):
            for r in ref:
                yield from self.space_elements_in_value_ref(r)
        else:
            yield from self._space_elements_in_atomic_value_ref(ref)

    def _space_elements_in_atomic_value_ref(
        self,
        ref: irefs.AtomicValueRef,
    ) -> Iterable[irefs.SpaceElementRef]:
        if isinstance(ref, irefs.IndexedRef):
            yield from self._space_elements_in_atomic_value_ref(ref.ref)
        else:
            yield ref
            space_ref = self.spaces[ref.space].space
            yield from self._space_elements_in_space_ref(space_ref)

    def _space_elements_in_space_ref(
        self, ref: irefs.SpaceRef
    ) -> Iterable[irefs.SpaceElementRef]:
        for a in ref.args:
            yield from self.space_elements_in_value_ref(a)


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
            entry per nested tree, which maps the id of the inducing
            space to the nested tree id.
        local_spaces: maps a node identifier to the identifiers of the
            local spaces defined in that node.
        query_answers: maps a space identifier denoting a query to the
            identifier of the answers provided for that query.

    """

    children: dict[irefs.NodeId, dict[irefs.ValueRef, irefs.NodeId]] = field(
        default_factory=lambda: defaultdict(lambda: {})
    )
    nested_trees: dict[irefs.NodeId, dict[irefs.SpaceId, irefs.NodeId]] = (
        field(default_factory=lambda: defaultdict(lambda: {}))
    )
    local_spaces: dict[irefs.NodeId, list[irefs.SpaceId]] = field(
        default_factory=lambda: defaultdict(lambda: [])
    )
    query_answers: dict[irefs.SpaceId, list[irefs.AnswerId]] = field(
        default_factory=lambda: defaultdict(lambda: [])
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
                case irefs.NestedIn(space_id):
                    parent_id = trace.spaces[space_id].node
                    map.nested_trees[parent_id][space_id] = child_id
        for space_id, origin in trace.spaces.items():
            map.local_spaces[origin.node].append(space_id)
        for answer_id, located in trace.answers.items():
            map.query_answers[located.space].append(answer_id)
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
        with self.lock:
            self.trace.convert_global_space_path(query.ref)

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
