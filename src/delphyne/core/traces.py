"""
Exporting and importing traces.
"""

from dataclasses import dataclass

from delphyne.core import refs


@dataclass(frozen=True)
class Location:
    node: refs.GlobalNodeRef
    space: refs.SpaceRef | None


#####
##### Exportable traces
#####


type NodeOriginStr = str


@dataclass(frozen=True)
class ExportableAnswer:
    answer: str
    mode: str | None = None


@dataclass
class ExportableQueryInfo:
    node: int
    ref: str
    answers: dict[int, ExportableAnswer]


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
    """
    By convention, `0` is the id of the global origin.
    """

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
        if answer in self.answer_ids[origin]:
            return self.answer_ids[origin][answer]
        else:
            self._last_answer_id += 1
            id = refs.AnswerId(self._last_answer_id)
            self.answers[id] = (origin, answer)
            self.answer_ids[origin][answer] = id
            return id

    def convert_global_node_path(
        self, path: refs.GlobalNodePath
    ) -> refs.NodeId:
        assert False

    def convert_node_path(
        self, node: refs.NodeId, path: refs.NodePath
    ) -> refs.NodeId:
        assert False

    def convert_space_ref(
        self, node: refs.NodeId, ref: refs.SpaceRef
    ) -> refs.SpaceRef:
        """
        Convert an _explicit_ space reference into an _abbreviated_ one
        that features node and answer ids.
        """
        assert False

    def convert_value_ref(
        self, node: refs.NodeId, ref: refs.ValueRef
    ) -> refs.ValueRef:
        assert False

    def convert_space_element_ref(
        self, node: refs.NodeId, ref: refs.SpaceElementRef
    ) -> refs.SpaceElementRef:
        assert False
