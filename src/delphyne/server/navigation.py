"""
Resolving references within a trace.

In this initial version, selectors are ignored.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Never, Protocol, Sequence, cast

from delphyne.core import refs, trees
from delphyne.core.queries import AnyQuery, ParseError, Query
from delphyne.core.refs import (
    AnswerId,
    ChoiceArgRef,
    ChoiceRef,
    Hint,
    NodeId,
    ValueRef,
)
from delphyne.core.strategies import StrategyTree
from delphyne.core.tracing import (
    Outcome,
    QueryOrigin,
    Tracer,
    Value,
    value_ref,
)
from delphyne.core.trees import Choice, Node, Success, Tree
from delphyne.utils.typing import NoTypeInfo, TypeAnnot


"""
Resolving References.
"""


type AnyNavTree = NavigationTree[Any, Any]


@dataclass
class HintReverseMap:
    actions: dict[tuple[NodeId, ValueRef], Sequence[Hint]]
    answers: dict[AnswerId, Hint | None]

    def __init__(self):
        self.actions = {}
        self.answers = {}


@dataclass
class NavigationInfo:
    """
    An object that is mutated during navigation to emit warnings.
    """

    hints_rev: HintReverseMap = field(default_factory=HintReverseMap)
    unused_hints: list[Hint] = field(default_factory=list)


class HintResolver(Protocol):
    def __call__(
        self,
        query: Query[Never, object],
        origin: QueryOrigin,
        hint: refs.HintStr | None,
    ) -> str | None:
        """
        Take a query and a hint and return a suitable answer. If no
        hint is provided, the default answer is expected.
        """
        ...


@dataclass
class Stuck(Exception):
    tree: AnyNavTree
    choice_ref: ChoiceRef
    remaining_hints: Sequence[Hint]


@dataclass
class ReachedFailureNode(Exception):
    tree: AnyNavTree
    remaining_hints: Sequence[Hint]


@dataclass
class Interrupted(Exception):
    tree: AnyNavTree
    remaining_hints: Sequence[Hint]


@dataclass
class AnswerParseError(Exception):
    query: AnyQuery
    answer: str
    error: str


@dataclass
class InvalidSubchoice(Exception):
    tree: AnyNavTree
    choice_label: str


@dataclass
class RefResolver:
    """
    Packages all information necessary to resolve references.
    """

    tracer: Tracer
    resolve_node_id: Callable[[NodeId], AnyNavTree]
    answer_from_hint: HintResolver | None = None
    info: NavigationInfo | None = None
    interrupt: Callable[[AnyNavTree], bool] | None = None

    def resolve_choice_arg_ref(
        self, tree: AnyNavTree, ref: ChoiceArgRef
    ) -> trees.ChoiceArg:
        if isinstance(ref, int):
            return ref
        return self.resolve_value_ref(tree, ref)

    def resolve_value_ref(self, tree: AnyNavTree, ref: refs.ValueRef) -> Value:
        if isinstance(ref, refs.ChoiceOutcomeRef):
            return self.resolve_choice_outcome_ref(tree, ref)
        return tuple(self.resolve_value_ref(tree, r) for r in ref)

    def resolve_subchoice_ref(
        self, tree: AnyNavTree, ref: refs.ChoiceRef
    ) -> Choice[object]:
        label, arg_refs = ref
        args = [self.resolve_choice_arg_ref(tree, r) for r in arg_refs]
        return _node_subchoice(tree, label, *args)

    def resolve_choice_outcome_ref(
        self, tree: AnyNavTree, ref: refs.ChoiceOutcomeRef
    ) -> Outcome[object]:
        if ref.choice is None:
            choice_ref = _primary_choice_ref(tree)
        else:
            choice_ref = ref.choice
        choice = self.resolve_subchoice_ref(tree, choice_ref)
        match ref.value:
            case AnswerId():
                _, answer = self.tracer.answers[ref.value]
                query = choice.source()
                assert isinstance(query, Query)
                parsed = query.parse_answer(answer)
                if isinstance(parsed, ParseError):
                    raise AnswerParseError(query, answer, parsed.error)
                return Outcome(parsed, ref, query.return_type())
            case NodeId():
                success = self.resolve_node_id(ref.value)
                node = cast(Success[object], success.node)
                assert isinstance(success.node, Success)
                return node.success
            case refs.Hints():
                hints = ref.value.hints
                outcome, rem = self.outcome_from_hints(tree, choice, hints)
                if self.info is not None:
                    self.info.unused_hints += rem
                return outcome

    def outcome_from_hints(
        self, tree: AnyNavTree, choice: Choice[object], hints: Sequence[Hint]
    ) -> tuple[Outcome[object], Sequence[Hint]]:
        source = choice.source()
        origin = choice.get_origin()
        match source:
            case Query():
                return self.query_outcome_from_hints(
                    tree, source, origin, hints
                )
            case trees.StrategyInstance():
                spawned = tree.spawn(source.strategy, origin)
                final, hints = self.follow_hints(spawned, hints)
                success = cast(Success[Any], final.node)
                assert isinstance(success, Success)
                return success.success, hints
            case trees.FiniteChoice():
                assert False

    def query_outcome_from_hints(
        self,
        tree: AnyNavTree,
        query: Query[Any, Any],
        origin: ChoiceRef,
        hints: Sequence[Hint],
    ) -> tuple[Outcome[object], Sequence[Hint]]:
        used_hint: Hint | None = None
        # We start by declaring the query in case it is not already
        query_origin = QueryOrigin(tree.node_id, origin)
        self.tracer.declare_query(query_origin)
        # Then we can start
        assert self.answer_from_hint is not None
        # We figure out whether the hint is active
        hint_active = hints and (
            (sel := hints[0].query_name) is None or sel == query.name()
        )
        # We first try the provided hint if it is active
        if hint_active:
            answer = self.answer_from_hint(query, query_origin, hints[0].hint)
            if answer is not None:  # if the head hint was consumed
                used_hint = hints[0]
                hints = hints[1:]
        else:
            answer = None
        # If we could not use a hint, we try with the default answer
        if answer is None:
            answer = self.answer_from_hint(query, query_origin, None)
        # If we still have no answer, we're stuck
        if answer is None:
            raise Stuck(tree, origin, hints)
        parsed = query.parse_answer(answer)
        if isinstance(parsed, ParseError):
            raise AnswerParseError(query, answer, parsed.error)
        answer_id = self.tracer.fresh_or_cached_answer_id(answer, query_origin)
        outcome = Outcome(
            parsed,
            refs.ChoiceOutcomeRef(origin, answer_id),
            query.return_type(),
        )
        # If necessary, we contribute to the hint reverse map
        if self.info is not None:
            self.info.hints_rev.answers[answer_id] = used_hint
        return outcome, hints

    def action_from_hints(
        self, tree: AnyNavTree, hints: Sequence[Hint]
    ) -> tuple[Value, Sequence[Hint]]:
        original_hints = hints
        navigator = tree.node.navigate()
        try:
            choice = next(navigator)
            while True:
                outcome, hints = self.outcome_from_hints(tree, choice, hints)
                choice = navigator.send(outcome)
        except StopIteration as e:
            value = cast(Value, e.value)
            if self.info is not None:
                used_hints = original_hints[: len(original_hints) - len(hints)]
                key = (tree.node_id, value_ref(value))
                self.info.hints_rev.actions[key] = used_hints
            return value, hints

    def follow_hints(
        self, tree: AnyNavTree, hints: Sequence[Hint]
    ) -> tuple[AnyNavTree, Sequence[Hint]]:
        if tree.node.leaf_node():
            if isinstance(tree.node, Success):
                return tree, hints
            else:
                raise ReachedFailureNode(tree, hints)
        if self.interrupt is not None and self.interrupt(tree):
            raise Interrupted(tree, hints)
        value, hints = self.action_from_hints(tree, hints)
        return self.follow_hints(tree.child(value), hints)


def _node_subchoice(
    tree: AnyNavTree, name: refs.ChoiceLabel, *args: trees.ChoiceArg
) -> Choice[object]:
    try:
        attr = getattr(tree.node, name)
        return attr if not args else attr(*args)
    except TypeError:
        raise InvalidSubchoice(tree, name)


def _primary_choice_ref(tree: AnyNavTree) -> ChoiceRef:
    node = tree.node
    assert isinstance(node, Node)
    choice = node.primary_choice()
    assert choice is not None, f"Node {node.type_name()} has no primary choice"
    return choice.get_origin()


"""
Navigation trees.
"""


@dataclass
class TraceReverseMap:
    # Useful to optimize calls to _child
    children: dict[refs.NodeId, dict[refs.ValueRef, refs.NodeId]] = field(
        default_factory=lambda: defaultdict(lambda: {}))  # fmt: skip
    subtrees: dict[refs.NodeId, dict[refs.ChoiceRef, refs.NodeId]] = field(
        default_factory=lambda: defaultdict(lambda: {}))  # fmt: skip

    @staticmethod
    def make(tracer: Tracer) -> "TraceReverseMap":
        map = TraceReverseMap()
        for child_id, origin in tracer.nodes.items():
            match origin:
                case refs.ChildOf(parent_id, action):
                    map.children[parent_id][action] = child_id
                case refs.SubtreeOf(parent_id, choice):
                    map.subtrees[parent_id][choice] = child_id
        return map


@dataclass(frozen=True)
class NavigationTree[N: Node, T](Tree[N, T]):
    """
    A tree wrapper that caches all the nodes it visits.
    """

    tree: StrategyTree[N, T]
    _cache: dict[refs.NodeId, AnyNavTree]
    _rev_map: TraceReverseMap

    def __post_init__(self):
        self._cache_current()

    def _cache_current(self):
        self._cache[self.tree.node_id] = self

    @staticmethod
    def make[M: Node, U](tree: StrategyTree[M, U]):
        return NavigationTree(tree, {}, TraceReverseMap())

    @property
    def node(self) -> N | Success[T]:
        return self.tree.node

    @property
    def node_id(self) -> NodeId:
        return self.tree.node_id

    @property
    def tracer(self) -> Tracer:
        return self.tree.tracer

    def return_type(self) -> TypeAnnot[T] | NoTypeInfo:
        return self.tree.return_type()

    def child(self, action: Value) -> "NavigationTree[N, T]":
        tree = None
        aref = value_ref(action)
        self_id = self.tree.node_id
        if (cid := self._rev_map.children[self_id].get(aref)) is not None:
            if (next := self._cache.get(cid)) is not None:
                return next
        tree = self.tree.child(action)
        self._rev_map.children[self_id][aref] = tree.node_id
        return NavigationTree(tree, self._cache, self._rev_map)

    def spawn[L: Node, U](
        self, strategy: trees.StrategyComp[L, U], origin: ChoiceRef
    ) -> "NavigationTree[L, U]":
        spawned = None
        self_id = self.tree.node_id
        if (cid := self._rev_map.subtrees[self_id].get(origin)) is not None:
            if (next := self._cache.get(cid)) is not None:
                return next
        spawned = self.tree.spawn(strategy, origin)
        self._rev_map.subtrees[self_id][origin] = spawned.node_id
        return NavigationTree[L, U](spawned, self._cache, self._rev_map)

    def basic_resolver(self) -> RefResolver:
        return RefResolver(self.tree.tracer, self._goto_cached)

    def _goto_cached(self, node_id: NodeId) -> AnyNavTree:
        return self._cache[node_id]

    def goto(self, node_id: NodeId) -> AnyNavTree:
        if node_id in self._cache:
            return self._goto_cached(node_id)
        assert node_id != Tracer.ROOT_ID
        match self.tree.tracer.nodes[node_id]:
            case refs.ChildOf(parent_id, action):
                parent = self.goto(parent_id)
                resolver = self.basic_resolver()
                return parent.child(resolver.resolve_value_ref(parent, action))
            case refs.SubtreeOf(parent_id, choice_ref):
                parent = self.goto(parent_id)
                resolver = self.basic_resolver()
                choice = resolver.resolve_subchoice_ref(parent, choice_ref)
                source = choice.source()
                assert isinstance(source, trees.StrategyInstance)
                return parent.spawn(source.strategy, choice_ref)
