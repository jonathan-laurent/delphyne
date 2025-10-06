"""
Resolving id-based references in traces.
"""

from typing import Any, Literal, assert_type

import delphyne.core as dp
from delphyne.core import irefs


class IRefResolver:
    """
    Utility class for resolving id-based references in traces.
    """

    def __init__(
        self,
        trace: dp.Trace,
        root: dp.AnyTree | None = None,
        enable_caching: bool = True,
    ) -> None:
        self.trace = trace
        self.root = root
        self.cache: dict[irefs.NodeId, dp.AnyTree] | None = None
        if enable_caching:
            self.cache = {}

    def load_tree_cache(self, cache: dp.TreeCache) -> None:
        if self.cache is None:
            return
        for ref, tree in cache.items():
            id = self.trace.convert_global_node_ref(ref)
            self.cache[id] = tree

    def resolve_answer(self, ref: irefs.AnswerId) -> dp.Answer:
        return self.trace.answers[ref].answer

    def resolve_node(self, ref: irefs.NodeId) -> dp.AnyTree:
        if self.cache is not None and ref in self.cache:
            return self.cache[ref]
        origin = self.trace.nodes[ref]
        if isinstance(origin, irefs.ChildOf):
            parent = self.resolve_node(origin.node)
            action = self.resolve_value(origin.node, origin.action)
            tree = parent.child(action)
        else:
            assert_type(origin, irefs.NestedIn)
            space = self.resolve_space(origin.space)
            if space == "main":
                if self.root is None:
                    raise ValueError("IRefResolver.root is not set.")
                tree = self.root
            else:
                source = space.source()
                assert isinstance(source, dp.NestedTree)
                tree = source.spawn_tree()
        if self.cache is not None:
            self.cache[ref] = tree
        return tree

    def resolve_space(
        self, space_id: irefs.SpaceId
    ) -> dp.Space[Any] | Literal["main"]:
        space_def = self.trace.spaces[space_id]
        if isinstance(space_def, irefs.MainSpace):
            return "main"
        return self.resolve_space_def(space_def.node, space_def.space)

    def resolve_space_def(
        self, node: irefs.NodeId, ref: irefs.SpaceRef
    ) -> dp.Space[Any]:
        parent = self.resolve_node(node)
        assert not isinstance(parent.node, dp.Success)
        space_args = tuple(self.resolve_value(node, v) for v in ref.args)
        space = parent.node.nested_space(ref.name, space_args)
        assert space is not None
        return space

    def resolve_space_element(
        self, ref: irefs.SpaceElementRef
    ) -> dp.Tracked[Any]:
        space = self.resolve_space(ref.space)
        space_source = space.source() if space != "main" else None
        match space_source:
            case dp.NestedTree() | None:
                assert isinstance(ref.element, irefs.NodeId)
                tree = self.resolve_node(ref.element)
                assert isinstance(tree.node, dp.Success)
                return tree.node.success
            case dp.AttachedQuery():
                assert isinstance(ref.element, irefs.AnswerId)
                answer = self.resolve_answer(ref.element)
                parsed = space_source.parse_answer(answer)
                assert not isinstance(parsed, dp.ParseError)
                return parsed

    def resolve_atomic_value(
        self, node: irefs.NodeId, ref: irefs.AtomicValueRef
    ) -> dp.Tracked[Any]:
        match ref:
            case irefs.IndexedRef():
                container = self.resolve_atomic_value(node, ref.ref)
                return container[ref.index]
            case irefs.SpaceElementRef():
                return self.resolve_space_element(ref)

    def resolve_value(
        self, node: irefs.NodeId, ref: irefs.ValueRef
    ) -> dp.Value:
        match ref:
            case None:
                return None
            case tuple():
                return tuple(self.resolve_value(node, v) for v in ref)
            case _:
                return self.resolve_atomic_value(node, ref)
