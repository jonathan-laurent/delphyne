"""
Resolving references within a trace.
"""

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import delphyne.core as dp
import delphyne.core.demos as dm
from delphyne.core import AnyTree, refs

VAL_HINT_PREFIX = "#"


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
    keep track of information necessary to generate good feedback.
    """

    hints_rev: HintReverseMap = field(default_factory=HintReverseMap)
    unused_hints: list[refs.Hint] = field(default_factory=list[refs.Hint])


class HintResolver(Protocol):
    def __call__(
        self,
        query: dp.AttachedQuery[Any],
        hint: refs.HintValue | None,
        implicit_answer: Callable[[], str] | None,
    ) -> refs.Answer | None:
        """
        Take a query and a hint and return a suitable answer. If no hint
        is provided, the default answer is expected. If the hint cannot
        be resolved (e.g. the query is not in the demo or the hint is
        not included as an answer label), `None` should be returned.
        """
        ...


class IdentifierResolver(Protocol):
    def resolve_node(self, id: refs.NodeId) -> dp.AnyTree: ...

    def resolve_answer(self, id: refs.AnswerId) -> dp.Answer: ...


@dataclass
class EncounteredTags:
    node_tags: dict[str, int] = field(
        default_factory=lambda: defaultdict(lambda: 0)
    )
    space_tags: dict[str, int] = field(
        default_factory=lambda: defaultdict(lambda: 0)
    )


def _tag_selectors_match(
    selectors: dm.TagSelectors,
    tags: Sequence[str],
    encountered: dict[str, int],
):
    for clause in selectors:
        num = clause.num if clause.num is not None else 1
        tag = clause.tag
        if not (tag in tags and encountered[tag] == num):
            return False
    return True


#####
##### Navigator Exceptions
#####


@dataclass
class Stuck(Exception):
    tree: AnyTree
    space_ref: refs.SpaceRef
    remaining_hints: Sequence[refs.Hint]


@dataclass
class ReachedFailureNode(Exception):
    tree: AnyTree
    remaining_hints: Sequence[refs.Hint]


@dataclass
class MatchedSelector(Exception):
    tree: AnyTree
    remaining_hints: Sequence[refs.Hint]


@dataclass
class AnswerParseError(Exception):
    tree: AnyTree
    query: dp.AttachedQuery[Any]
    answer: dp.Answer
    error: dp.ParseError


@dataclass
class InvalidSpace(Exception):
    tree: AnyTree
    space_name: refs.SpaceName


@dataclass
class NoPrimarySpace(Exception):
    tree: AnyTree


#####
##### Navigator
#####


@dataclass
class Navigator:
    """
    Packages all information necessary to navigate a tree and resolve
    hint-based or id-based references.
    """

    hint_resolver: HintResolver | None = None
    id_resolver: IdentifierResolver | None = None
    info: NavigationInfo | None = None
    tracer: dp.Tracer | None = None

    def resolve_atomic_value_ref(
        self, tree: AnyTree, ref: refs.AtomicValueRef
    ) -> dp.Tracked[Any]:
        if isinstance(ref, refs.SpaceElementRef):
            return self.resolve_space_element_ref(tree, ref)
        else:
            parent = self.resolve_atomic_value_ref(tree, ref.ref)
            return parent[ref.index]

    def resolve_value_ref(self, tree: AnyTree, ref: refs.ValueRef) -> dp.Value:
        if ref is None:
            return None
        elif isinstance(ref, tuple):
            return tuple(self.resolve_value_ref(tree, r) for r in ref)
        else:
            return self.resolve_atomic_value_ref(tree, ref)

    def resolve_space_ref(
        self, tree: AnyTree, ref: refs.SpaceRef
    ) -> dp.Space[Any]:
        args = tuple(self.resolve_value_ref(tree, r) for r in ref.args)
        space = tree.node.nested_space(ref.name, args)
        if space is None:
            raise InvalidSpace(tree, ref.name)
        return space

    def resolve_space_element_ref(
        self, tree: AnyTree, ref: refs.SpaceElementRef
    ) -> dp.Tracked[Any]:
        # Use the primary space if no space name is provided.
        if ref.space is None:
            space_ref = tree.node.primary_space_ref()
            if space_ref is None:
                raise NoPrimarySpace(tree)
        else:
            space_ref = ref.space
        space = self.resolve_space_ref(tree, space_ref)
        match ref.element:
            case refs.HintsRef():
                hints = ref.element.hints
                elt, rem = self.space_element_from_hints(
                    tree, space, hints, None, EncounteredTags()
                )
                if self.info is not None:
                    self.info.unused_hints += rem
                return elt
            case refs.AnswerId():
                assert self.id_resolver is not None
                ans = self.id_resolver.resolve_answer(ref.element)
                query = space.source()
                assert isinstance(query, dp.AttachedQuery)
                parsed = query.parse_answer(ans)
                if isinstance(parsed, dp.ParseError):
                    raise AnswerParseError(tree, query, ans, parsed)
                return parsed
            case refs.NodeId():
                assert self.id_resolver is not None
                success = self.id_resolver.resolve_node(ref.element)
                assert isinstance(success.node, dp.Success)
                return success.node.success
            case _:
                assert False

    def space_element_from_hints(
        self,
        tree: AnyTree,
        space: dp.Space[Any],
        hints: Sequence[refs.Hint],
        selector: dm.NodeSelector | None,
        encountered: EncounteredTags,
    ) -> tuple[dp.Tracked[Any], Sequence[refs.Hint]]:
        source = space.source()
        for tag in space.tags():
            encountered.space_tags[tag] += 1
        match source:
            case dp.AttachedQuery():
                implicit = None
                if isinstance(tree.node, dp.ComputationNode):
                    implicit = tree.node.run_computation
                return self.answer_from_hints(tree, source, hints, implicit)
            case dp.NestedTree():
                tree = source.spawn_tree()
                # New selector
                sub_selector: dm.NodeSelector | None = None
                if isinstance(
                    selector, dm.WithinSpace
                ) and _tag_selectors_match(
                    selector.space, space.tags(), encountered.space_tags
                ):
                    sub_selector = selector.selector
                final, hints = self.follow_hints(
                    tree, hints, sub_selector, EncounteredTags()
                )
                # `follow_hints` raises an exception if a success is not
                # reached.
                assert isinstance(final.node, dp.Success)
                return final.node.success, hints

    def follow_hints(
        self,
        tree: AnyTree,
        hints: Sequence[refs.Hint],
        selector: dm.NodeSelector | None,
        encountered: EncounteredTags,
    ) -> tuple[AnyTree, Sequence[refs.Hint]]:
        if tree.node.leaf_node():
            if isinstance(tree.node, dp.Success):
                return tree, hints
            else:
                raise ReachedFailureNode(tree, hints)
        # We register that the tags were encountered
        tags = tree.node.get_tags()
        for tag in tags:
            encountered.node_tags[tag] += 1
        # We see test whether we've reached our target node
        if selector and not isinstance(selector, dm.WithinSpace):
            if _tag_selectors_match(selector, tags, encountered.node_tags):
                raise MatchedSelector(tree, hints)
        value, hints = self.action_from_hints(
            tree, hints, selector, encountered
        )
        return self.follow_hints(
            tree.child(value), hints, selector, encountered
        )

    def action_from_hints(
        self,
        tree: AnyTree,
        hints: Sequence[refs.Hint],
        selector: dm.NodeSelector | None,
        encountered: EncounteredTags,
    ) -> tuple[dp.Value, Sequence[refs.Hint]]:
        original_hints = hints
        try:
            navigator = tree.node.navigate()
            space = next(navigator)
            while True:
                elt, hints = self.space_element_from_hints(
                    tree, space, hints, selector, encountered
                )
                space = navigator.send(elt)
        except StopIteration as e:
            value = cast(dp.Value, e.value)
            if self.info is not None:
                used_hints = original_hints[: len(original_hints) - len(hints)]
                key = (tree.ref, refs.value_ref(value))
                self.info.hints_rev.actions[key] = used_hints
            return value, hints

    def answer_from_hints(
        self,
        tree: AnyTree,
        query: dp.AttachedQuery[Any],
        hints: Sequence[refs.Hint],
        implicit: Callable[[], str] | None,
    ) -> tuple[dp.Tracked[Any], Sequence[refs.Hint]]:
        assert self.hint_resolver is not None
        if self.tracer is not None:
            self.tracer.trace_query(query.ref)
        # TODO: we ignore qualifiers because they should not work this way.
        used_hint: refs.Hint | None = None
        answer: refs.Answer | None = None
        # We first try the first hint if there is one
        if hints:
            hint = hints[0].hint
            if hint.startswith(VAL_HINT_PREFIX):
                # The hint is a value hint
                if (
                    sel_ans := _find_answer_in_finite_set(
                        query, hint.removeprefix(VAL_HINT_PREFIX)
                    )
                ) is not None:
                    answer = sel_ans
            else:
                # We don't send the implicit answer the first time since
                # we don't want to consume the hint.
                answer = self.hint_resolver(query, hints[0].hint, None)
            if answer is not None:
                used_hint = hints[0]
                hints = hints[1:]
        # If we could not use a hint, we try with the default answer
        if answer is None:
            answer = self.hint_resolver(query, None, implicit)
        # If we still cannot, resolve, maybe the query has a default answer
        if answer is None:
            if (defa := query.query.default_answer()) is not None:
                answer = defa
        # If we still have no answer, we're stuck
        if answer is None:
            raise Stuck(tree, query.ref[1], hints)
        parsed = query.parse_answer(answer)
        if isinstance(parsed, dp.ParseError):
            raise AnswerParseError(tree, query, answer, parsed)
        # We contribute to the hint reverse map
        if self.info is not None:
            self.info.hints_rev.answers[(query.ref, answer)] = used_hint
        return parsed, hints


def _find_answer_in_finite_set(
    query: dp.AttachedQuery[Any], content: str
) -> refs.Answer | None:
    aset = query.query.finite_answer_set()
    if not aset:
        return None
    for a in aset:
        if isinstance(a.content, str) and a.content == content:
            return a
    return None
