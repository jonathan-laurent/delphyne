"""
Resolving references within a trace.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol, cast

import delphyne.core as dp
from delphyne.core import refs

#####
##### Navigation Trees
#####


@dataclass(frozen=True)
class NavTree:
    """
    A tree wrapper that caches all the nodes it visits.
    """

    tree: dp.Tree[Any, Any, Any]
    _cache: "dict[refs.GlobalNodePath, NavTree]"
    _spaces_cache: dict[refs.GlobalSpacePath, dp.Space[Any]]

    def __post_init__(self):
        self._cache_current()

    def _cache_current(self):
        self._cache[self.tree.ref] = self

    @staticmethod
    def make(tree: dp.Tree[Any, Any, Any]) -> "NavTree":
        assert tree.ref == refs.MAIN_ROOT
        return NavTree(tree, {}, {})

    @property
    def node(self) -> dp.Node | dp.Success[Any]:
        return self.tree.node

    @property
    def ref(self) -> refs.GlobalNodePath:
        return self.tree.ref

    def child(self, action: dp.Value) -> "NavTree":
        aref = refs.value_ref(action)
        cref = refs.child_ref(self.ref, aref)
        if cref in self._cache:
            return self._cache[cref]
        return replace(self, tree=self.tree.child(action))

    def nested_space(
        self, space_name: refs.SpaceName, args: tuple[dp.Value, ...]
    ) -> dp.Space[Any] | None:
        arg_refs = tuple(refs.value_ref(arg) for arg in args)
        sref = refs.SpaceRef(space_name, arg_refs)
        gsref = (self.ref, sref)
        if gsref in self._spaces_cache:
            return self._spaces_cache[gsref]
        space = self.node.nested_space(space_name, args)
        if space is None:
            return None
        self._spaces_cache[gsref] = space
        return space

    def _convert_nested_tree(
        self, nested: dp.NestedTree[Any, Any, Any]
    ) -> "NavTree":
        assert nested.ref[0] == self.ref
        nref = refs.nested_ref(self.ref, nested.ref[1])
        if nref in self._cache:
            return self._cache[nref]
        return replace(self, tree=nested.spawn_tree())

    def space_source(
        self, space: dp.Space[Any]
    ) -> "NavTree | dp.AttachedQuery[Any]":
        source = space.source()
        if isinstance(source, dp.NestedTree):
            return self._convert_nested_tree(source)
        return source

    def goto(self, ref: refs.GlobalNodePath) -> "NavTree":
        """
        Go to a node from its id, assuming it is in the cache (which it
        should be if it was reached before, and it must have been
        reached before since navigation trees can only be spawned at the
        main root -- see `NavTree.make`).
        """
        return self._cache[ref]

    def goto_space(self, ref: refs.GlobalSpacePath) -> dp.Space[Any]:
        return self._spaces_cache[ref]


#####
##### Navigator Utilities
#####


type ActionRef = tuple[refs.GlobalNodePath, refs.ValueRef]
type AnswerRef = tuple[refs.GlobalSpacePath, refs.Answer]


@dataclass
class HintReverseMap:
    """
    When visualizing the trace associated to a demo, it is convenient to
    have references that use hints rather than ids (e.g. `compare(['',
    'foo bar'])` instead of `compare([cands{%2}, cands{%3}])`). However,
    rewriting all references with hints is nontrivial and for it to be
    possible, we keep track during navigation of how every answer and
    action was computed from hints.
    """

    actions: dict[ActionRef, Sequence[refs.Hint]]
    answers: dict[AnswerRef, refs.Hint | None]

    def __init__(self):
        self.actions = {}
        self.answers = {}


@dataclass
class NavigationInfo:
    """
    An object that is mutated during navigation to emit warnings and
    keep track of information neecssary to generate good feedback.
    """

    hints_rev: HintReverseMap = field(default_factory=HintReverseMap)
    unused_hints: list[refs.Hint] = field(default_factory=list)


class HintResolver(Protocol):
    def __call__(
        self,
        query: dp.AttachedQuery[Any],
        hint: refs.HintValue | None,
    ) -> refs.Answer | None:
        """
        Take a query and a hint and return a suitable answer. If no hint
        is provided, the default answer is expected. If the hint cannot
        be resolved (e.g. the query is not in the demo or the hint is
        not included as an answer label), `None` should be returned.
        """
        ...


#####
##### Navigator Exceptions
#####


@dataclass
class Stuck(Exception):
    tree: NavTree
    choice_ref: refs.SpaceRef
    remaining_hints: Sequence[refs.Hint]


@dataclass
class ReachedFailureNode(Exception):
    tree: NavTree
    remaining_hints: Sequence[refs.Hint]


@dataclass
class Interrupted(Exception):
    tree: NavTree
    remaining_hints: Sequence[refs.Hint]


@dataclass
class AnswerParseError(Exception):
    tree: NavTree
    query: dp.AttachedQuery[Any]
    answer: str
    error: str


@dataclass
class InvalidSpace(Exception):
    tree: NavTree
    space_name: refs.SpaceName


#####
##### Navigator
#####


@dataclass
class Navigator:
    """
    Packages all information necessary to navigate a tree and resolve
    hint-based references.
    """

    hint_resolver: HintResolver
    info: NavigationInfo | None = None
    interrupt: Callable[[NavTree], bool] | None = None

    def resolve_value_ref(self, tree: NavTree, ref: refs.ValueRef) -> dp.Value:
        if isinstance(ref, refs.SpaceElementRef):
            return self.resolve_space_element_ref(tree, ref)
        return tuple(self.resolve_value_ref(tree, r) for r in ref)

    def resolve_space_ref(
        self, tree: NavTree, ref: refs.SpaceRef
    ) -> NavTree | dp.AttachedQuery[Any]:
        args = tuple(self.resolve_value_ref(tree, r) for r in ref.args)
        space = tree.nested_space(ref.name, args)
        if space is None:
            raise InvalidSpace(tree, ref.name)
        return tree.space_source(space)

    def resolve_space_element_ref(
        self, tree: NavTree, ref: refs.SpaceElementRef
    ) -> dp.Tracked[Any]:
        # Use the primary space if no space name is provided.
        if ref.space is None:
            space_ref = tree.node.primary_space_ref()
            assert (
                space_ref is not None
            ), f"Node {tree.node.effect_name()} has no primary space"
        else:
            space_ref = ref.space
        space = self.resolve_space_ref(tree, space_ref)
        # Only hint-based references are supported.
        assert isinstance(ref.element, refs.Hints)
        hints = ref.element.hints
        elt, rem = self.space_element_from_hints(tree, space, hints)
        if self.info is not None:
            self.info.unused_hints += rem
        return elt

    def space_element_from_hints(
        self,
        tree: NavTree,
        space: dp.AttachedQuery[Any] | NavTree,
        hints: Sequence[refs.Hint],
    ) -> tuple[dp.Tracked[Any], Sequence[refs.Hint]]:
        match space:
            case dp.AttachedQuery():
                return self.answer_from_hints(tree, space, hints)
            case NavTree():
                final, hints = self.follow_hints(space, hints)
                success = cast(dp.Success[Any], final.node)
                assert isinstance(success, dp.Success)
                return success.success, hints

    def follow_hints(
        self, tree: NavTree, hints: Sequence[refs.Hint]
    ) -> tuple[NavTree, Sequence[refs.Hint]]:
        if tree.node.leaf_node():
            if isinstance(tree.node, dp.Success):
                return tree, hints
            else:
                raise ReachedFailureNode(tree, hints)
        if self.interrupt is not None and self.interrupt(tree):
            raise Interrupted(tree, hints)
        value, hints = self.action_from_hints(tree, hints)
        return self.follow_hints(tree.child(value), hints)

    def action_from_hints(
        self, tree: NavTree, hints: Sequence[refs.Hint]
    ) -> tuple[dp.Value, Sequence[refs.Hint]]:
        original_hints = hints
        navigator = tree.node.navigate()
        try:
            space = tree.space_source(next(navigator))
            while True:
                elt, hints = self.space_element_from_hints(tree, space, hints)
                space = tree.space_source(navigator.send(elt))
        except StopIteration as e:
            value = cast(dp.Value, e.value)
            if self.info is not None:
                used_hints = original_hints[: len(original_hints) - len(hints)]
                key = (tree.ref, refs.value_ref(value))
                self.info.hints_rev.actions[key] = used_hints
            return value, hints

    def answer_from_hints(
        self,
        tree: NavTree,
        query: dp.AttachedQuery[Any],
        hints: Sequence[refs.Hint],
    ) -> tuple[dp.Tracked[Any], Sequence[refs.Hint]]:
        # TODO: we ignore selectors because they should not work this way.
        used_hint: refs.Hint | None = None
        # We first try the first hint if there is one
        if hints:
            answer = self.hint_resolver(query, hints[0].hint)
            if answer is not None:
                used_hint = hints[0]
                hints = hints[1:]
        else:
            answer = None
        # If we could not use a hint, we try with the default answer
        if answer is None:
            answer = self.hint_resolver(query, None)
        # If we still have no answer, we're stuck
        if answer is None:
            raise Stuck(tree, query.ref[1], hints)
        parsed = query.answer(answer.mode, answer.text)
        if isinstance(parsed, dp.ParseError):
            raise AnswerParseError(tree, query, answer.text, parsed.error)
        # We contribute to the hint reverse map
        if self.info is not None:
            self.info.hints_rev.answers[(query.ref, answer)] = used_hint
        return parsed, hints
