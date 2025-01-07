"""
Generating Browsable Traces.
"""

import pprint
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import delphyne.core as dp
from delphyne.analysis import feedback as fb
from delphyne.analysis import navigation as nv
from delphyne.core import demos as dm
from delphyne.core import refs
from delphyne.utils import typing as tp

#####
##### Reference Simplification
#####


@dataclass
class _RefSimplifier:
    """
    Transforms standard references into hint-based references.
    """

    tree: nv.NavTree
    hint_rev_map: nv.HintReverseMap

    def action(
        self, node: refs.GlobalNodePath, action: refs.ValueRef
    ) -> Sequence[refs.Hint] | None:
        return self.hint_rev_map.actions.get((node, action))

    def path_to(
        self, src_id: refs.GlobalNodePath, dst_id: refs.GlobalNodePath
    ) -> Sequence[refs.Hint] | None:
        # Compute a sequence of hints necessary to go from the source
        # node __or the root of a directly nested tree__ to the destination.
        if src_id == dst_id:
            return ()
        match refs.global_path_origin(dst_id):
            case "global_origin":
                assert False
            case ("nested", dst_origin, _):
                if dst_origin == src_id:
                    return ()
                return None
            case ("child", before, action):
                action_hints = self.hint_rev_map.actions.get((before, action))
                if action_hints is None:
                    return None
                prefix = self.path_to(src_id, before)
                if prefix is None:
                    return None
                return tuple([*prefix] + [*action_hints])

    def space_element_ref(
        self, id: refs.GlobalNodePath, ref: refs.SpaceElementRef
    ) -> refs.SpaceElementRef:
        assert ref.space is not None
        # We start by converting the element answer ref or success path
        match ref.element:
            case refs.Hints() | refs.AnswerId() | refs.NodeId():
                assert False
            case refs.Answer():
                aref: nv.AnswerRef = ((id, ref.space), ref.element)
                if aref in self.hint_rev_map.answers:
                    hint = self.hint_rev_map.answers[aref]
                    hints = refs.Hints((hint,) if hint is not None else ())
                    ref = refs.SpaceElementRef(ref.space, hints)
            case tuple():  # Node path
                gpath: refs.GlobalNodePath = (*id, (ref.space, ref.element))
                hints_raw = self.path_to(id, gpath)
                assert hints_raw is not None
                hints = tuple(hints_raw)
                ref = refs.SpaceElementRef(ref.space, refs.Hints(hints))
        # We make the space reference implicit if we can ('foo bar'
        # instead of cands{'foo bar'}).
        assert isinstance(ref.element, refs.Hints)
        tree = self.tree.goto(id)
        node = tree.node
        assert isinstance(node, dp.Node)
        primary_ref = node.primary_space_ref()
        if primary_ref is not None and ref.space == primary_ref:
            ref = refs.SpaceElementRef(None, ref.element)
        elif ref.space is not None:
            # Else we recursively simplify the space reference.
            ref = refs.SpaceElementRef(
                self.space_ref(id, ref.space), ref.element
            )
        return ref

    def value_ref(
        self, id: refs.GlobalNodePath, v: refs.ValueRef
    ) -> refs.ValueRef:
        if isinstance(v, tuple):
            return tuple(self.value_ref(id, v) for v in v)
        else:
            return self.space_element_ref(id, v)

    def space_ref(
        self, id: refs.GlobalNodePath, ref: refs.SpaceRef
    ) -> refs.SpaceRef:
        args = tuple(self.value_ref(id, a) for a in ref.args)
        return refs.SpaceRef(ref.name, args)


#####
##### Representing General Python Objects
#####


def _check_valid_json(obj: object) -> bool:
    match obj:
        case int() | float() | str() | bool() | None:
            return True
        case dict():
            obj = cast(dict[object, object], obj)
            return all(
                isinstance(k, str) and _check_valid_json(v)
                for k, v in obj.items()
            )
        case tuple() | list():
            obj = cast(Sequence[object], obj)
            return all(_check_valid_json(v) for v in obj)
        case _:
            return False


def _value_repr[T](
    obj: T, typ: tp.TypeAnnot[T] | tp.NoTypeInfo
) -> fb.ValueRepr:
    short = pprint.pformat(obj, compact=True, sort_dicts=False)
    long = pprint.pformat(obj, compact=False, sort_dicts=False)
    value = fb.ValueRepr(short, long, False, None)
    if not isinstance(typ, tp.NoTypeInfo):
        try:
            json = tp.pydantic_dump(typ, obj)
            assert _check_valid_json(json)
            value.json = json
            value.json_provided = True
        except Exception:
            pass
    return value


#####
##### Listing all local spaces and elements
#####


## Enumerating spaces


def _spaces_in_node_origin(
    origin: refs.NodeOrigin,
) -> Iterable[refs.SpaceRef]:
    """
    From a `Trace`, we want to recover a list of all the spawned spaces
    for every encountered node. For this, we compile a list of all local
    space references.
    """
    match origin:
        case refs.ChildOf():
            yield from _spaces_in_value_ref(origin.action)
        case refs.NestedTreeOf():
            yield from _spaces_in_space_ref(origin.space)


def _spaces_in_value_ref(
    value: refs.ValueRef,
) -> Iterable[refs.SpaceRef]:
    if isinstance(value, refs.SpaceElementRef):
        if value.space is not None:
            yield from _spaces_in_space_ref(value.space)
    else:
        for v in value:
            yield from _spaces_in_value_ref(v)


def _spaces_in_space_ref(ref: refs.SpaceRef) -> Iterable[refs.SpaceRef]:
    yield ref
    for a in ref.args:
        yield from _spaces_in_value_ref(a)


## Enumerating space elements


def _space_elements_in_value_ref(
    value: refs.ValueRef,
) -> Iterable[refs.SpaceElementRef]:
    if isinstance(value, refs.SpaceElementRef):
        yield value
        if value.space is not None:
            yield from _space_elements_in_space_ref(value.space)
    else:
        for v in value:
            yield from _space_elements_in_value_ref(v)


def _space_elements_in_space_ref(
    ref: refs.SpaceRef,
) -> Iterable[refs.SpaceElementRef]:
    for a in ref.args:
        yield from _space_elements_in_value_ref(a)


#####
##### Browsable Traces
#####


def compute_browsable_trace(
    tree: nv.NavTree, simplifier: _RefSimplifier | None = None
) -> fb.Trace:
    return _TraceTranslator(tree, simplifier).translate_trace()
