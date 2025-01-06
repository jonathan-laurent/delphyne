"""
Exporting and importing traces.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from delphyne.core import pprint, refs


@dataclass(frozen=True)
class Location:
    node: refs.GlobalNodeRef
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
            case refs.Hints():
                assert False
            case tuple():
                element = self.convert_node_path(id, ref.element)
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

    def convert_global_node_ref(self, ref: refs.GlobalNodeRef) -> refs.NodeId:
        if isinstance(ref, refs.NodeId):
            return ref
        else:
            return self.convert_global_node_path(ref)

    def add_location(self, location: Location) -> None:
        id = self.convert_global_node_ref(location.node)
        if location.space is not None:
            self.convert_space_ref(id, location.space)

    def add_locations(self, locations: Iterable[Location]) -> None:
        for location in locations:
            self.add_location(location)

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
