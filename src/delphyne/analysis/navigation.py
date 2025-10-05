"""
Resolving references within a trace.
"""

from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, cast

import delphyne.core as dp
import delphyne.core.demos as dm
from delphyne.core import AnyTree, hrefs, refs

VAL_HINT_PREFIX = "#"


#####
##### Navigator Utilities
#####


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

    actions: dict[refs.GlobalValueRef, Sequence[hrefs.Hint]]
    answers: dict[refs.GlobalAnswerRef, hrefs.Hint | None]

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
    unused_hints: list[hrefs.Hint] = field(default_factory=list[hrefs.Hint])


class HintResolver(ABC):
    """
    An oracle for answering queries, with or without hints.
    """

    @abstractmethod
    def answer_with_hint(
        self, query: dp.AttachedQuery[Any], hint: hrefs.HintValue
    ) -> refs.Answer | None:
        """
        Try and answer a query using a hint.

        If the hint is not applicable, return None.
        """
        pass

    @abstractmethod
    def answer_without_hint(
        self, query: dp.AttachedQuery[Any], tree: AnyTree
    ) -> refs.Answer | None:
        """
        Try and answer a query without a hint.

        Arguments:
            query: The query to answer.
            tree: The tree that the query belongs to. This information
                may be useful for generating implicit answers (see
                `ImplicitAnswerGenerator`).
        """
        pass


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
    remaining_hints: Sequence[hrefs.Hint]


@dataclass
class ReachedFailureNode(Exception):
    tree: AnyTree
    remaining_hints: Sequence[hrefs.Hint]


@dataclass
class MatchedSelector(Exception):
    tree: AnyTree
    remaining_hints: Sequence[hrefs.Hint]


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
    info: NavigationInfo | None = None
    tracer: dp.Tracer | None = None

    def resolve_atomic_value_ref(
        self, tree: AnyTree, ref: hrefs.AtomicValueRef
    ) -> dp.Tracked[Any]:
        if isinstance(ref, hrefs.SpaceElementRef):
            return self.resolve_space_element_ref(tree, ref)
        else:
            parent = self.resolve_atomic_value_ref(tree, ref.ref)
            return parent[ref.index]

    def resolve_value_ref(
        self, tree: AnyTree, ref: hrefs.ValueRef
    ) -> dp.Value:
        if ref is None:
            return None
        elif isinstance(ref, tuple):
            return tuple(self.resolve_value_ref(tree, r) for r in ref)
        else:
            return self.resolve_atomic_value_ref(tree, ref)

    def resolve_space_ref(
        self, tree: AnyTree, ref: hrefs.SpaceRef
    ) -> dp.Space[Any]:
        args = tuple(self.resolve_value_ref(tree, r) for r in ref.args)
        space = tree.node.nested_space(ref.name, args)
        if space is None:
            raise InvalidSpace(tree, ref.name)
        return space

    def resolve_space_element_ref(
        self, tree: AnyTree, ref: hrefs.SpaceElementRef
    ) -> dp.Tracked[Any]:
        # Use the primary space if no space name is provided.
        if ref.space is None:
            pspace_ref = tree.node.primary_space_ref()
            if pspace_ref is None:
                raise NoPrimarySpace(tree)
            space_ref = hrefs.convert_trivial_space_ref(pspace_ref)
        else:
            space_ref = ref.space
        space = self.resolve_space_ref(tree, space_ref)
        hints = ref.element
        elt, rem = self.space_element_from_hints(
            tree, space, hints, None, EncounteredTags()
        )
        if self.info is not None:
            self.info.unused_hints += rem
        return elt

    def space_element_from_hints(
        self,
        tree: AnyTree,
        space: dp.Space[Any],
        hints: Sequence[hrefs.Hint],
        selector: dm.NodeSelector | None,
        encountered: EncounteredTags,
    ) -> tuple[dp.Tracked[Any], Sequence[hrefs.Hint]]:
        source = space.source()
        for tag in space.tags():
            encountered.space_tags[tag] += 1
        match source:
            case dp.AttachedQuery():
                return self.answer_from_hints(tree, source, hints)
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
        hints: Sequence[hrefs.Hint],
        selector: dm.NodeSelector | None,
        encountered: EncounteredTags,
    ) -> tuple[AnyTree, Sequence[hrefs.Hint]]:
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
        hints: Sequence[hrefs.Hint],
        selector: dm.NodeSelector | None,
        encountered: EncounteredTags,
    ) -> tuple[dp.Value, Sequence[hrefs.Hint]]:
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
        hints: Sequence[hrefs.Hint],
    ) -> tuple[dp.Tracked[Any], Sequence[hrefs.Hint]]:
        assert self.hint_resolver is not None
        if self.tracer is not None:
            self.tracer.trace_query(query)
        # TODO: we ignore qualifiers because they should not work this way.
        used_hint: hrefs.Hint | None = None
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
                answer = self.hint_resolver.answer_with_hint(
                    query, hints[0].hint
                )
            if answer is not None:
                used_hint = hints[0]
                hints = hints[1:]
        # If we could not use a hint, we try with the default answer
        if answer is None:
            answer = self.hint_resolver.answer_without_hint(query, tree)
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
