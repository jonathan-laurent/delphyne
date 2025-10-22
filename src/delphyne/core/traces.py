"""
Representing, exporting and importing traces.

A trace (`Trace`) denotes a collection of reachable nodes and spaces,
which is encoded in a concise way by introducing unique identifiers for
answers and nodes.
"""

import threading
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeGuard

from delphyne.core import irefs, parse, refs
from delphyne.core.trees import AttachedQuery, Tree
from delphyne.utils.typing import pydantic_dump

type Location = refs.GlobalNodeRef | refs.GlobalSpacePath | None
"""
Optional location information for log messages.

Log messages can be attached to a given node or space.
"""

type ShortLocation = irefs.NodeId | irefs.SpaceId | None
"""
Optional location information for exportable log messages.
"""

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
class ExportableLocatedAnswer:
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
    answers: dict[int, ExportableLocatedAnswer]


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

    MAIN_SPACE_ID = irefs.SpaceId(0)

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

        self.spaces[Trace.MAIN_SPACE_ID] = irefs.MainSpace()
        self.space_ids[irefs.MainSpace()] = Trace.MAIN_SPACE_ID

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
            id.id: ExportableLocatedAnswer(located.space.id, located.answer)
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
            id_bis = self.convert_global_node_ref(expanded)
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

    def convert_global_space_path(
        self, ref: refs.GlobalSpacePath
    ) -> irefs.SpaceId:
        """
        Convert a full, global space reference denoting a quey origin
        into an id-based reference.
        """
        id = Trace.MAIN_SPACE_ID
        for path, space in ref.steps:
            nid = self.fresh_or_cached_node_id(irefs.NestedIn(id))
            nid = self._convert_node_path(nid, path)
            id = self._convert_space_ref(nid, space)
        return id

    def convert_global_node_ref(
        self, path: refs.GlobalNodeRef
    ) -> irefs.NodeId:
        """
        Convert a full, global node reference into an id-based one.
        """
        space_id = self.convert_global_space_path(path.space)
        root = self.fresh_or_cached_node_id(irefs.NestedIn(space_id))
        return self._convert_node_path(root, path.path)

    def convert_answer_ref(self, ref: refs.GlobalAnswerRef) -> irefs.AnswerId:
        """
        Convert a full answer reference into an answer id.
        """
        space = self.convert_global_space_path(ref[0])
        located = irefs.LocatedAnswer(space, ref[1])
        return self.fresh_or_cached_answer_id(located)

    def convert_location(self, location: Location) -> ShortLocation:
        """
        Convert a full location into an id-based one.
        """
        match location:
            case None:
                return None
            case refs.GlobalNodeRef():
                return self.convert_global_node_ref(location)
            case refs.GlobalSpacePath():
                return self.convert_global_space_path(location)

    def _convert_space_ref(
        self, node: irefs.NodeId, ref: refs.SpaceRef
    ) -> irefs.SpaceId:
        """
        Convert a full local space reference into an id-based one, relative
        to a given node.
        """
        args = tuple(self._convert_value_ref(node, a) for a in ref.args)
        space_ref = irefs.SpaceRef(ref.name, args)
        return self.fresh_or_cached_space_id(irefs.LocalSpace(node, space_ref))

    def _convert_node_path(
        self, node: irefs.NodeId, path: refs.NodePath
    ) -> irefs.NodeId:
        """
        Convert a full local node path into an identifier, relative to a
        given node.
        """
        for a in path.actions:
            action_ref = self._convert_value_ref(node, a)
            node = self.fresh_or_cached_node_id(
                irefs.ChildOf(node, action_ref)
            )
        return node

    def _convert_atomic_value_ref(
        self, node: irefs.NodeId, ref: refs.AtomicValueRef
    ) -> irefs.AtomicValueRef:
        """
        Convert a full local atomic value reference into an id-based one,
        relative to a given node.
        """
        if isinstance(ref, refs.IndexedRef):
            return irefs.IndexedRef(
                self._convert_atomic_value_ref(node, ref.ref), ref.index
            )
        else:
            return self.convert_space_element_ref(node, ref)

    def _convert_value_ref(
        self, node: irefs.NodeId, ref: refs.ValueRef
    ) -> irefs.ValueRef:
        """
        Convert a full local value reference into an id-based one,
        relative to a given node.
        """
        if ref is None:
            return None
        elif isinstance(ref, tuple):
            return tuple(self._convert_value_ref(node, a) for a in ref)
        else:
            return self._convert_atomic_value_ref(node, ref)

    def convert_space_element_ref(
        self, node: irefs.NodeId, ref: refs.SpaceElementRef
    ) -> irefs.SpaceElementRef:
        """
        Convert a full local space element reference into an id-based one,
        relative to a given node.
        """
        # We leverage locality to speed up the computation.
        # The following would work but be much slower:
        #     space_id = self.convert_global_space_path(ref.space)

        # The space is attached to a node with an identifier.
        assert ref.space is not None
        space_id = self._convert_space_ref(node, ref.space)
        match ref.element:
            case refs.Answer():
                located = irefs.LocatedAnswer(space_id, ref.element)
                element = self.fresh_or_cached_answer_id(located)
            case refs.NodePath():
                nested_root_orig = irefs.NestedIn(space_id)
                nested_root = self.fresh_or_cached_node_id(nested_root_orig)
                element = self._convert_node_path(nested_root, ref.element)
        return irefs.SpaceElementRef(space_id, element)

    ### Reverse direction: expanding id-based references into full ones.

    def expand_global_space_id(
        self, id: irefs.SpaceId
    ) -> refs.GlobalSpacePath:
        rev_steps: list[tuple[refs.NodePath, refs.SpaceRef]] = []
        origin = self.spaces[id]
        while not isinstance(origin, irefs.MainSpace):
            space_ref = self.expand_space_ref(origin.space)
            id, path = self._recover_path(origin.node)
            rev_steps.append((path, space_ref))
            origin = self.spaces[id]
        return refs.GlobalSpacePath(tuple(reversed(rev_steps)))

    def _recover_path(
        self, dst: irefs.NodeId
    ) -> tuple[irefs.SpaceId, refs.NodePath]:
        """
        Find the space from which the tree containing `dst` originates,
        along with the path from the root of that tree to `dst`.
        """
        rev_path: list[refs.ValueRef] = []
        while True:
            dst_origin = self.nodes[dst]
            match dst_origin:
                case irefs.ChildOf(before, action):
                    rev_path.append(self.expand_value_ref(action))
                    dst = before
                case irefs.NestedIn(space_id):
                    path = refs.NodePath(tuple(reversed(rev_path)))
                    return (space_id, path)

    def _expand_located_answer_ref(
        self, ans: irefs.LocatedAnswer
    ) -> refs.GlobalAnswerRef:
        space = self.expand_global_space_id(ans.space)
        return (space, ans.answer)

    def expand_answer_id(self, ans: irefs.AnswerId) -> refs.GlobalAnswerRef:
        located = self.answers[ans]
        return self._expand_located_answer_ref(located)

    def expand_node_id(self, id: irefs.NodeId) -> refs.GlobalNodeRef:
        """
        Convert a node identifier into a full, global node reference.
        """
        orig, path = self._recover_path(id)
        space = self.expand_global_space_id(orig)
        return refs.GlobalNodeRef(space, path)

    def expand_space_ref(self, ref: irefs.SpaceRef) -> refs.SpaceRef:
        """
        Convert a local id-based space reference into a full one,
        relative to a given node.
        """
        args = tuple(self.expand_value_ref(a) for a in ref.args)
        return refs.SpaceRef(ref.name, args)

    def expand_value_ref(self, ref: irefs.ValueRef) -> refs.ValueRef:
        """
        Convert a local id-based value reference into a full one,
        relative to a given node.
        """
        if ref is None:
            return None
        elif isinstance(ref, tuple):
            return tuple(self.expand_value_ref(a) for a in ref)
        else:
            return self._expand_atomic_value_ref(ref)

    def _expand_atomic_value_ref(
        self, ref: irefs.AtomicValueRef
    ) -> refs.AtomicValueRef:
        """
        Convert a local id-based atomic value reference into a full one,
        relative to a given node.
        """
        if isinstance(ref, irefs.IndexedRef):
            return refs.IndexedRef(
                self._expand_atomic_value_ref(ref.ref), ref.index
            )
        else:
            return self._expand_space_element_ref(ref)

    def _expand_space_element_ref(
        self, ref: irefs.SpaceElementRef
    ) -> refs.SpaceElementRef:
        """
        Convert a local id-based space element reference into a full
        one, relative to a given node.
        """
        # The following would work but be terribly inefficient:
        #    space = self.expand_global_space_id(ref.space)
        space_def = self.spaces[ref.space]
        assert not isinstance(space_def, irefs.MainSpace)
        local_space = self.expand_space_ref(space_def.space)
        match ref.element:
            case irefs.AnswerId():
                located = self.answers[ref.element]
                element = located.answer
            case irefs.NodeId():
                space_id, element = self._recover_path(ref.element)
                assert space_id == ref.space
        return refs.SpaceElementRef(local_space, element)

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
            space_def = self.spaces[ref.space]
            if isinstance(space_def, irefs.LocalSpace):
                yield from self._space_elements_in_space_ref(space_def.space)

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
                    space_def = trace.spaces[space_id]
                    if isinstance(space_def, irefs.LocalSpace):
                        parent_id = space_def.node
                        map.nested_trees[parent_id][space_id] = child_id
        for space_id, space_def in trace.spaces.items():
            if isinstance(space_def, irefs.LocalSpace):
                map.local_spaces[space_def.node].append(space_id)
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
        id: Optionally, a unique identifier for the message, which can
            be used to tie related messages together.
        related: Optionally, a list of identifiers of related messages.
    """

    message: str
    level: LogLevel
    time: datetime
    metadata: object | None = None
    location: ShortLocation | None = None
    id: str | None = None
    related: Sequence[str] | None = None


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
    space: int | None = None
    metadata: object | None = None  # JSON value
    id: str | None = None
    related: Sequence[str] | None = None


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

    def global_node_id(self, node: refs.GlobalNodeRef) -> irefs.NodeId:
        """
        Ensure that a node at a given reference is present in the trace
        and return the corresponding node identififier.
        """
        with self.lock:
            return self.trace.convert_global_node_ref(node)

    def trace_node(self, node: refs.GlobalNodeRef) -> None:
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
        id: str | None = None,
        related: Sequence[str] | None = None,
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
                    id=id,
                    related=related,
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
                if isinstance(m.location, irefs.NodeId):
                    node = m.location.id
                if isinstance(m.location, irefs.SpaceId):
                    space = m.location.id
                yield ExportableLogMessage(
                    message=m.message,
                    level=m.level,
                    time=m.time if not remove_timing_info else None,
                    node=node,
                    space=space,
                    metadata=pydantic_dump(object, m.metadata),
                    id=m.id,
                    related=m.related,
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


def generate_unique_log_message_id() -> str:
    """
    Return a unique random identifier with 6 characters, which can be
    used to tie log messages together.
    """

    # Generate a uuid and take a prefix
    import uuid

    return str(uuid.uuid4())[:6]
