"""
Automatically keeping track of provenance info when manipulating trees.
"""

from dataclasses import dataclass
from typing import Any

from delphyne.core import parse, pprint, refs
from delphyne.core.refs import AnswerId, Assembly, NodeId, NodeOrigin, ValueRef
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


@dataclass(frozen=True)
class Outcome[T]:
    value: T
    ref: refs.ChoiceOutcomeRef
    type: TypeAnnot[T] | NoTypeInfo


type Value = Assembly[Outcome[object]]


def value_ref(v: Value) -> ValueRef:
    match v:
        case Outcome(_, ref):
            return ref
        case tuple():
            return tuple(value_ref(o) for o in v)


def value_type(v: Value) -> TypeAnnot[Any] | NoTypeInfo:
    match v:
        case Outcome(_, _, type):
            return type
        case tuple():
            types = [value_type(o) for o in v]
            if any(isinstance(t, NoTypeInfo) for t in types):
                return NoTypeInfo()
            return tuple[*types]  # type: ignore


def drop_refs(v: Value) -> object:
    match v:
        case Outcome(value, _):
            return value
        case tuple():
            return tuple(drop_refs(o) for o in v)


type NodeOriginStr = str


@dataclass(frozen=True)
class QueryOrigin:
    node: NodeId
    ref: refs.ChoiceRef


@dataclass
class ExportableQueryInfo:
    node: int
    ref: str
    answers: dict[int, str]


@dataclass
class ExportableTrace:
    """
    A lightweight trace format that can be easily exported to JSON/YAML.
    """

    nodes: dict[int, NodeOriginStr]
    queries: list[ExportableQueryInfo]


class Tracer:
    """
    Utility for keeping track of all the visited nodes of a tree, export
    it and reconstruct it.

    - Success paths are not used in traces. Hints are only used to index
      finite choices.
    """

    ROOT_ID = NodeId(0)
    ROOT_ORIGIN = ("__main__", ())

    def __init__(self):
        self.nodes: dict[NodeId, NodeOrigin] = {}
        self.node_ids: dict[NodeOrigin, NodeId] = {}
        self.answers: dict[AnswerId, tuple[QueryOrigin, str]] = {}
        self.answer_ids: dict[QueryOrigin, dict[str, AnswerId]] = {}
        self._last_node_id: int = 0
        self._last_answer_id: int = 0

    def fresh_or_cached_node_id(self, origin: NodeOrigin) -> NodeId:
        assert refs.basic_node_origin(origin)
        if origin in self.node_ids:
            return self.node_ids[origin]
        else:
            self._last_node_id += 1
            id = NodeId(self._last_node_id)
            self.nodes[id] = origin
            self.node_ids[origin] = id
            return id

    def declare_query(self, origin: QueryOrigin) -> None:
        if origin not in self.answer_ids:
            self.answer_ids[origin] = {}

    def fresh_or_cached_answer_id(
        self, value: str, origin: QueryOrigin
    ) -> AnswerId:
        assert origin in self.answer_ids
        if value in self.answer_ids[origin]:
            return self.answer_ids[origin][value]
        else:
            self._last_answer_id += 1
            id = AnswerId(self._last_answer_id)
            self.answers[id] = (origin, value)
            self.answer_ids[origin][value] = id
            return id

    def export(self) -> ExportableTrace:
        nodes = {
            id.id: pprint.node_origin(origin)
            for id, origin in self.nodes.items()
        }
        queries: list[ExportableQueryInfo] = []
        for q, a in self.answer_ids.items():
            ref = pprint.choice_ref(q.ref)
            answers = {id.id: value for value, id in a.items()}
            queries.append(ExportableQueryInfo(q.node.id, ref, answers))
        return ExportableTrace(nodes, queries)

    @staticmethod
    def load(trace: ExportableTrace) -> "Tracer":
        tracer = Tracer()
        for id, origin_str in trace.nodes.items():
            node_id = NodeId(id)
            origin = parse.node_origin(origin_str)
            tracer.nodes[node_id] = origin
            tracer.node_ids[origin] = node_id
        for q in trace.queries:
            origin = QueryOrigin(NodeId(q.node), parse.choice_ref(q.ref))
            tracer.answer_ids[origin] = {}
            for a, v in q.answers.items():
                aid = AnswerId(a)
                tracer.answers[aid] = (origin, v)
                tracer.answer_ids[origin][v] = aid
        tracer._last_node_id = max(*trace.nodes.keys())
        tracer._last_answer_id = max(id.id for id in tracer.answers.keys())
        return tracer
