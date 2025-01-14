"""
Exporting and importing traces.
"""

from collections import defaultdict
from dataclasses import dataclass, field

from delphyne.core import pprint, refs


@dataclass(frozen=True)
class Location:
    node: refs.GlobalNodePath
    space: refs.SpaceRef | None


@dataclass(frozen=True)
class ShortLocation:
    node: refs.NodeId
    space: refs.SpaceRef | None


#####
##### Exportable traces
#####


type NodeOriginStr = str


@dataclass
class ExportableQueryInfo:
    node: int
    space: str
    answers: dict[int, refs.Answer]


@dataclass
class ExportableTrace:
    """
    A lightweight trace format that can be easily exported to JSON/YAML.
    """

    nodes: dict[int, NodeOriginStr]
    queries: list[ExportableQueryInfo]


#####
##### Traces
#####


@dataclass(frozen=True)
class QueryOrigin:
    node: refs.NodeId
    ref: refs.SpaceRef


class Trace:
    GLOBAL_ORIGIN_ID = refs.NodeId(0)

    def __init__(self):
        self.nodes: dict[refs.NodeId, refs.NodeOrigin] = {}
        self.node_ids: dict[refs.NodeOrigin, refs.NodeId] = {}
        self.answers: dict[refs.AnswerId, tuple[QueryOrigin, refs.Answer]] = {}
        self.answer_ids: dict[
            QueryOrigin, dict[refs.Answer, refs.AnswerId]
        ] = {}
        self._last_node_id: int = 0
        self._last_answer_id: int = 0

    def fresh_or_cached_node_id(self, origin: refs.NodeOrigin) -> refs.NodeId:
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

    # Convert full references into id-based references, registering ids
    # on the fly as needed.

    def convert_node_path(
        self, id: refs.NodeId, path: refs.NodePath
    ) -> refs.NodeId:
        for a in path:
            action_ref = self.convert_value_ref(id, a)
            id = self.fresh_or_cached_node_id(refs.ChildOf(id, action_ref))
        return id

    def convert_space_ref(
        self, id: refs.NodeId, ref: refs.SpaceRef
    ) -> refs.SpaceRef:
        """
        Convert an _explicit_ space reference into an _abbreviated_ one
        that features node and answer ids.
        """
        args = tuple(self.convert_value_ref(id, a) for a in ref.args)
        return refs.SpaceRef(ref.name, args)

    def convert_value_ref(
        self, id: refs.NodeId, ref: refs.ValueRef
    ) -> refs.ValueRef:
        if isinstance(ref, refs.SpaceElementRef):
            return self.convert_space_element_ref(id, ref)
        else:
            return tuple(self.convert_value_ref(id, a) for a in ref)

    def convert_answer_ref(
        self, ref: tuple[refs.GlobalSpacePath, refs.Answer]
    ) -> refs.AnswerId:
        node_path, space = ref[0]
        id = self.convert_global_node_path(node_path)
        space = self.convert_space_ref(id, space)
        origin = QueryOrigin(id, space)
        return self.fresh_or_cached_answer_id(ref[1], origin)

    def convert_space_element_ref(
        self, id: refs.NodeId, ref: refs.SpaceElementRef
    ) -> refs.SpaceElementRef:
        space = None
        if ref.space is not None:
            space = self.convert_space_ref(id, ref.space)
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
                element = self.convert_node_path(nested_root, ref.element)
        return refs.SpaceElementRef(space, element)

    def convert_global_node_path(
        self, path: refs.GlobalNodePath
    ) -> refs.NodeId:
        id = Trace.GLOBAL_ORIGIN_ID
        for space, node_path in path:
            space_ref = self.convert_space_ref(id, space)
            id = self.fresh_or_cached_node_id(refs.NestedTreeOf(id, space_ref))
            id = self.convert_node_path(id, node_path)
        return id

    def convert_global_space_path(
        self, path: refs.GlobalSpacePath
    ) -> refs.SpaceRef:
        node_path, space_ref = path
        id = self.convert_global_node_path(node_path)
        return self.convert_space_ref(id, space_ref)

    # Reverse direction: expand id-based references into full ones.
    # For now, fail with an assertion error if the id-based ref is invalid.

    def recover_path(
        self, dst: refs.NodeId
    ) -> tuple[refs.NodeId, refs.SpaceRef, refs.NodePath]:
        # Find the node from which the tree containing `dst` originates.
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

    def expand_space_ref(
        self, id: refs.NodeId, ref: refs.SpaceRef
    ) -> refs.SpaceRef:
        args = tuple(self.expand_value_ref(id, a) for a in ref.args)
        return refs.SpaceRef(ref.name, args)

    def expand_value_ref(
        self, id: refs.NodeId, ref: refs.ValueRef
    ) -> refs.ValueRef:
        if isinstance(ref, refs.SpaceElementRef):
            return self.expand_space_element_ref(id, ref)
        else:
            return tuple(self.expand_value_ref(id, a) for a in ref)

    def expand_space_element_ref(
        self, id: refs.NodeId, ref: refs.SpaceElementRef
    ) -> refs.SpaceElementRef:
        assert ref.space is not None
        space = self.expand_space_ref(id, ref.space)
        match ref.element:
            case refs.AnswerId():
                _orig, ans = self.answers[ref.element]
                element = ans
            case refs.NodeId():
                orig, _, element = self.recover_path(ref.element)
                assert orig == id
            case _:
                assert False
        return refs.SpaceElementRef(space, element)

    def expand_node_id(self, id: refs.NodeId) -> refs.GlobalNodePath:
        rev_path: list[tuple[refs.SpaceRef, refs.NodePath]] = []
        while id != Trace.GLOBAL_ORIGIN_ID:
            id, space, path = self.recover_path(id)
            rev_path.append((space, path))
        return tuple(reversed(rev_path))

    # Utilities

    def convert_location(self, location: Location) -> ShortLocation:
        id = self.convert_global_node_path(location.node)
        space = None
        if location.space is not None:
            space = self.convert_space_ref(id, location.space)
        return ShortLocation(id, space)

    def export(self) -> ExportableTrace:
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
        for id in self.nodes:
            expanded = self.expand_node_id(id)
            assert id == self.convert_global_node_path(expanded)


#####
##### Reverse Map
#####


@dataclass
class TraceReverseMap:
    # Useful to optimize calls to _child
    children: dict[refs.NodeId, dict[refs.ValueRef, refs.NodeId]] = field(
        default_factory=lambda: defaultdict(lambda: {}))  # fmt: skip
    nested_trees: dict[refs.NodeId, dict[refs.SpaceRef, refs.NodeId]] = field(
        default_factory=lambda: defaultdict(lambda: {}))  # fmt: skip

    @staticmethod
    def make(trace: Trace) -> "TraceReverseMap":
        map = TraceReverseMap()
        for child_id, origin in trace.nodes.items():
            match origin:
                case refs.ChildOf(parent_id, action):
                    map.children[parent_id][action] = child_id
                case refs.NestedTreeOf(parent_id, space):
                    map.nested_trees[parent_id][space] = child_id
        return map
