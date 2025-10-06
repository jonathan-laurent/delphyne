"""
Generating Browsable Traces.

Traces defined by `core.traces.Trace` contain all the information
necessary to recompute a trace but are not easily manipulated by tools.
In comparison, `analysis.feedback.Trace` contains a more redundant but
also more explicit view. This module provides a way to convert a trace
from the former format to the latter.
"""

import pprint
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import delphyne.core as dp
from delphyne.analysis import feedback as fb
from delphyne.analysis import navigation as nv
from delphyne.analysis.resolvers import IRefResolver
from delphyne.core import hrefs, irefs, refs
from delphyne.utils import typing as tp

#####
##### Reference Simplification
#####


@dataclass
class RefSimplifier:
    """
    Transforms standard references into hint-based references.

    A cache must be passed featuring all accessed nodes. The reason this
    cache is needed is that we call `primary_space_ref` on individual
    nodes to know whether or not space names can be elided.

    WARNING: the simplifies works with full references and not id-based
    ones. Use `Trace.expand_*` to convert if needed.
    """

    cache: dp.TreeCache
    hint_rev_map: nv.HintReverseMap

    def action(
        self, node: refs.GlobalNodeRef, action: refs.ValueRef
    ) -> Sequence[hrefs.Hint] | None:
        return self.hint_rev_map.actions.get((node, action))

    def path_to(
        self, space: refs.GlobalSpacePath, path: refs.NodePath
    ) -> tuple[hrefs.Hint, ...]:
        hints: list[hrefs.Hint] = []
        cur = refs.GlobalNodeRef(space, refs.NodePath(()))
        for action in path.actions:
            action_hints = self.action(cur, action)
            if action_hints is None:
                raise ValueError("RefSimplifier: incomplete hint_rev_map.")
            hints.extend(action_hints)
        return tuple(hints)

    def space_element_ref(
        self, node_ref: refs.GlobalNodeRef, ref: refs.SpaceElementRef
    ) -> hrefs.SpaceElementRef:
        assert ref.space is not None
        gsref = node_ref.nested_space(ref.space)
        match ref.element:
            case refs.Answer():
                aref: refs.GlobalAnswerRef = (gsref, ref.element)
                if aref not in self.hint_rev_map.answers:
                    raise ValueError("Incomplete rev_map.")
                hint = self.hint_rev_map.answers[aref]
                hints = (hint,) if hint is not None else ()
            case refs.NodePath():
                hints = self.path_to(gsref, ref.element)
        # We make the space reference implicit if we can ('foo bar'
        # instead of cands{'foo bar'}).
        tree = self.cache[node_ref]
        node = tree.node
        assert isinstance(node, dp.Node)
        primary_ref = node.primary_space_ref()
        if primary_ref is not None and ref.space == primary_ref:
            return hrefs.SpaceElementRef(None, hints)
        else:
            h_space_ref = self.space_ref(node_ref, ref.space)
            return hrefs.SpaceElementRef(h_space_ref, hints)

    def atomic_value_ref(
        self, id: refs.GlobalNodeRef, ref: refs.AtomicValueRef
    ) -> hrefs.AtomicValueRef:
        if isinstance(ref, refs.IndexedRef):
            parent = self.atomic_value_ref(id, ref.ref)
            return hrefs.IndexedRef(parent, ref.index)
        else:
            return self.space_element_ref(id, ref)

    def value_ref(
        self, id: refs.GlobalNodeRef, v: refs.ValueRef
    ) -> hrefs.ValueRef:
        if v is None:
            return None
        elif isinstance(v, tuple):
            return tuple(self.value_ref(id, a) for a in v)
        else:
            return self.atomic_value_ref(id, v)

    def space_ref(
        self, id: refs.GlobalNodeRef, ref: refs.SpaceRef
    ) -> hrefs.SpaceRef:
        args = tuple(self.value_ref(id, a) for a in ref.args)
        return hrefs.SpaceRef(ref.name, args)

    def answer(self, ref: refs.GlobalAnswerRef) -> hrefs.Hint | None:
        return self.hint_rev_map.answers[ref]


#####
##### Representing General Python Objects
#####


def _value_repr[T](
    obj: T, typ: tp.TypeAnnot[T] | tp.NoTypeInfo
) -> fb.ValueRepr:
    short = str(obj)
    # short = pprint.pformat(obj, compact=True, sort_dicts=False)
    long = pprint.pformat(obj, compact=False, sort_dicts=False)
    value = fb.ValueRepr(
        short=short, long=long, json_provided=False, json=None
    )
    if not isinstance(typ, tp.NoTypeInfo):
        try:
            json = tp.pydantic_dump(typ, obj)
            assert tp.valid_json_object(json)
            value.json = json
            value.json_provided = True
        except Exception:
            pass
    return value


#####
##### Browsable Traces
#####


def compute_browsable_trace(
    trace: dp.Trace,
    cache: dp.TreeCache,
    simplifier: RefSimplifier | None = None,
) -> fb.Trace:
    """
    Compute a browsable trace from a raw trace.

    A simplifier is typically only available for demonstrations.
    """
    resolver = IRefResolver(trace)
    resolver.load_tree_cache(cache)
    tr = _TraceTranslator(trace, resolver, simplifier)
    return tr.translate_trace()


class _TraceTranslator:
    def __init__(
        self,
        trace: dp.Trace,
        resolver: IRefResolver,
        simplifier: RefSimplifier | None = None,
    ) -> None:
        self.trace = trace
        self.resolver = resolver
        self.rev_map = dp.TraceReverseMap.make(trace)
        self.simplifier = simplifier
        # Each local space is mapped to a property id. Note that local
        # data can also consume such ids.
        self.space_prop_ids: dict[irefs.SpaceId, fb.TraceNodePropertyId] = {}
        # For each node, we map action references to ids.
        self.action_ids: dict[
            tuple[irefs.NodeId, irefs.ValueRef], fb.TraceActionId
        ] = {}
        # Both `self.space_prop_ids` and `self.action_ids` are populated
        # when a node is translated and read later by `translate_origin`
        # when processing a child node or a directly nested node.

    def translate_trace(self) -> fb.Trace:
        # We rely on the nodes in the trace being presented in
        # topological order (see explanation in `translate_node`).
        ids = list(self.trace.nodes.keys())
        trace = fb.Trace({id.id: self.translate_node(id) for id in ids})
        return trace

    def translate_node(self, id: irefs.NodeId) -> fb.Node:
        # Computing the success value if any
        tree = self.resolver.resolve_node(id)
        node = tree.node
        if isinstance(node, dp.Success):
            value = refs.drop_refs(node.success)
            success = _value_repr(value, node.success.type_annot)
        else:
            success = None
        # We populate `self.space_prop_ids`. It is important that nodes
        # are translated in topological order. Indeed, when a node is
        # processed and its origin is translated (`translate_origin`),
        # the property id of its originator is needed.
        local_spaces = self.rev_map.local_spaces[id]
        for i, space_id in enumerate(local_spaces):
            self.space_prop_ids[space_id] = i
        # TODO: add data fields.
        # Now we can translate properties
        props = [self.translate_space(sid) for sid in local_spaces]
        # Computing actions. The same reasoning than for property ids
        # applies since `translate_origin` also reads action ids.
        actions: list[fb.Action] = []
        for i, (a, dst) in enumerate(self.rev_map.children[id].items()):
            actions.append(self.translate_action(id, a, dst))
            self.action_ids[(id, a)] = i
        # Gather everything
        return fb.Node(
            kind=node.effect_name(),
            success_value=success,
            summary_message=node.summary_message(),
            tags=list(node.get_tags()),
            leaf_node=node.leaf_node(),
            label="&".join(ts) if (ts := node.get_tags()) else None,
            properties=props,
            actions=actions,
            origin=self.translate_origin(id),
        )

    def translate_strategy_comp(
        self,
        space_id: irefs.SpaceId,
        strategy: dp.StrategyComp[Any, Any, Any],
        tags: Sequence[dp.Tag],
    ) -> fb.NodeProperty:
        strategy_name = strategy.strategy_name()
        if strategy_name is None:
            strategy_name = "<anon>"
        # We get the strategy instance arguments and use typing hints to
        # render them properly.
        args_raw = strategy.strategy_arguments()
        hints = strategy.strategy_argument_types()
        args = {a: _value_repr(v, hints[a]) for a, v in args_raw.items()}
        # We obtain the root id of the nested tree, which can be `None`
        # if the tree hasn't been explored.
        origin = self.trace.spaces[space_id]
        assert not isinstance(origin, irefs.MainSpace)
        root_id = self.rev_map.nested_trees[origin.node].get(space_id)
        root_id_raw = root_id.id if root_id is not None else None
        return fb.NestedTree(
            kind="nested",
            strategy=strategy_name,
            args=args,
            tags=list(tags),
            node_id=root_id_raw,
        )

    def translate_query(
        self,
        space_id: irefs.SpaceId,
        query: dp.AbstractQuery[Any],
        tags: Sequence[dp.Tag],
    ) -> fb.NodeProperty:
        name = query.query_name()
        args = query.serialize_args()
        answers: list[fb.Answer] = []
        for aid in self.rev_map.query_answers.get(space_id, []):
            a = self.trace.answers[aid].answer
            parsed = query.parse_answer(a)
            parsed_repr = _value_repr(parsed, query.answer_type())
            hint_str: tuple[()] | tuple[str] | None = None
            if self.simplifier is not None:
                # If a simplifier is provided, the associated hint must
                # be in the rev map.
                full_aref = self.trace.expand_answer_id(aid)
                hint = self.simplifier.answer(full_aref)
                if hint is None:
                    hint_str = ()
                else:
                    hint_str = (hint.hint,)
            answers.append(
                fb.Answer(id=aid.id, hint=hint_str, value=parsed_repr)
            )
        return fb.Query(
            kind="query",
            name=name,
            args=args,
            tags=list(tags),
            answers=answers,
        )

    def translate_space(
        self, space_id: irefs.SpaceId
    ) -> tuple[fb.Reference, fb.NodeProperty]:
        space = self.resolver.resolve_space(space_id)
        assert space != "main"
        match source := space.source():
            case dp.NestedTree():
                prop = self.translate_strategy_comp(
                    space_id, source.strategy, space.tags()
                )
            case dp.AttachedQuery():
                prop = self.translate_query(
                    space_id, source.query, space.tags()
                )
        # In the UI, what do we want to show?
        origin = self.trace.spaces[space_id]
        assert not isinstance(origin, irefs.MainSpace)
        ref_str = fb.Reference(with_ids=str(origin.space), with_hints=None)
        if self.simplifier is not None:
            full_nref = self.trace.expand_node_id(origin.node)
            full_sref = self.trace.expand_space_ref(origin.space)
            simplified = self.simplifier.space_ref(full_nref, full_sref)
            ref_str.with_hints = str(simplified)
        return (ref_str, prop)

    def translate_action(
        self, src: irefs.NodeId, action: irefs.ValueRef, dst: irefs.NodeId
    ) -> fb.Action:
        # The `dst` argument is the id of the node that the action leads to
        # Compute two representations of the reference
        ref_str = fb.Reference(
            with_ids=irefs.show_value_ref(action), with_hints=None
        )
        hints_str: list[str] | None = None
        if self.simplifier is not None:
            # There are two ways actions could be shown in the UI using
            # hints. See `feedback.Action`.
            full_src_ref = self.trace.expand_node_id(src)
            full_aref = self.trace.expand_value_ref(action)
            hints = self.simplifier.action(full_src_ref, full_aref)
            simplified_value_ref = self.simplifier.value_ref(
                full_src_ref, full_aref
            )
            ref_str.with_hints = hrefs.show_value_ref(simplified_value_ref)
            if hints is None:
                hints_str = None
            else:
                hints_str = [str(h) for h in hints]
        # Computing related successes and answers (see `feedback`
        # documentation). Using dicts instead of sets to ensure determinism.
        related_successes: dict[fb.TraceNodeId, None] = {}
        related_answers: dict[fb.TraceAnswerId, None] = {}
        for elt in self.trace.space_elements_in_value_ref(action):
            if isinstance(elt.element, irefs.NodeId):
                related_successes[elt.element.id] = None
            else:
                related_answers[elt.element.id] = None
        # Rendering the value itself by resolving it.
        value = self.resolver.resolve_value(src, action)
        repr = _value_repr(refs.drop_refs(value), refs.value_type(value))
        return fb.Action(
            ref=ref_str,
            hints=hints_str,
            related_success_nodes=list(related_successes),
            related_answers=list(related_answers),
            value=repr,
            destination=dst.id,
        )

    def translate_origin(self, id: irefs.NodeId) -> fb.NodeOrigin:
        match self.trace.nodes[id]:
            case irefs.ChildOf(parent, action):
                action_id = self.action_ids[(parent, action)]
                return ("child", parent.id, action_id)
            case irefs.NestedIn(space_id):
                space_def = self.trace.spaces[space_id]
                if isinstance(space_def, irefs.MainSpace):
                    return "root"
                parent = space_def.node
                prop_id = self.space_prop_ids[space_id]
                return ("nested", parent.id, prop_id)
