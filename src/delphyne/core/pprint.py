"""
Pretty-printing references.

Paths and inline answers are not supported and so exceptions are raised
when these occur. In particular, full references cannot be
pretty-printed but id-based and hint-based references can.
"""

from collections.abc import Callable, Sequence
from typing import cast

from delphyne.core import demos, refs


class PathDetected(Exception): ...


class AnswerDetected(Exception): ...


NONE_REF_REPR = "nil"


def node_id(id: refs.NodeId) -> str:
    return f"%{id.id}"


def answer_id(id: refs.AnswerId) -> str:
    return f"@{id.id}"


def assembly[T](pp: Callable[[T], str], a: refs.Assembly[T]) -> str:
    """Print an assembly, assuming that T does not intersect with tuple."""
    if isinstance(a, tuple):
        a = cast(tuple[refs.Assembly[T], ...], a)
        return "[" + ", ".join(assembly(pp, x) for x in a) + "]"
    elif a is None:
        return NONE_REF_REPR
    else:
        return pp(a)


def space_name(space: refs.SpaceName) -> str:
    ret = space.name
    for i in space.indices:
        ret += f"[{i}]"
    return ret


def space_ref(sr: refs.SpaceRef) -> str:
    name = space_name(sr.name)
    if not sr.args:
        return name
    args_str = ", ".join(assembly(value_ref, a) for a in sr.args)
    return f"{name}({args_str})"


def atomic_value_ref(vr: refs.AtomicValueRef) -> str:
    if isinstance(vr, refs.IndexedRef):
        return f"{atomic_value_ref(vr.ref)}[{vr.index}]"
    return space_element_ref(vr)


def value_ref(vr: refs.ValueRef) -> str:
    return assembly(atomic_value_ref, vr)


def hint(h: refs.Hint) -> str:
    if not h.qualifier:
        return h.hint
    return f"{h.qualifier}:{h.hint}"


def hints(hs: Sequence[refs.Hint]):
    return "'" + " ".join(hint(h) for h in hs) + "'"


def answer(a: refs.Answer) -> str:
    ret = repr(a.content)
    if a.mode is not None:
        ret = f"{a.mode}:{ret}"
    return ret


def node_path(p: refs.NodePath) -> str:
    return "<" + ", ".join(value_ref(sr) for sr in p) + ">"


def space_element_ref(ser: refs.SpaceElementRef) -> str:
    if ser.space is None:
        assert isinstance(ser.element, refs.HintsRef)
        return hints(ser.element.hints)
    match ser.element:
        case refs.Answer():
            # raise AnswerDetected()
            value = answer(ser.element)
        case refs.AnswerId():
            value = answer_id(ser.element)
        case refs.NodeId():
            value = node_id(ser.element)
        case refs.HintsRef():
            value = hints(ser.element.hints)
        case _:
            assert isinstance(ser.element, tuple)
            # raise PathDetected()
            value = node_path(ser.element)
    return f"{space_ref(ser.space)}{{{value}}}"


def global_node_path(p: refs.GlobalNodePath) -> str:
    inner = "; ".join(f"{space_ref(sr)}, {node_path(np)}" for sr, np in p)
    return "<" + inner + ">"


def node_origin(no: refs.NodeOrigin) -> str:
    match no:
        case refs.ChildOf(node, action):
            return f"child({node.id}, {value_ref(action)})"
        case refs.NestedTreeOf(node, space):
            return f"nested({node.id}, {space_ref(space)})"


class CmdNames:
    RUN = "run"
    RUN_UNTIL = "at"
    SELECT = "go"
    GO_TO_CHILD = "take"
    ANSWER = "answer"
    IS_SUCCESS = "success"
    IS_FAILURE = "failure"
    SAVE = "save"
    LOAD = "load"


def tag_selector(selector: demos.TagSelector) -> str:
    ret = selector.tag
    if selector.num is not None:
        ret += "#" + str(selector.num)
    return ret


def tag_selectors(selectors: demos.TagSelectors) -> str:
    return "&".join(tag_selector(sel) for sel in selectors)


def node_selector(selector: demos.NodeSelector) -> str:
    if isinstance(selector, demos.WithinSpace):
        return (
            tag_selectors(selector.space)
            + "/"
            + node_selector(selector.selector)
        )
    else:
        return tag_selectors(selector)


def test_step(ts: demos.TestStep) -> str:
    match ts:
        case demos.Run(hs, None):
            if not hs:
                return CmdNames.RUN
            return f"{CmdNames.RUN} {hints(hs)}"
        case demos.Run(hs, until):
            assert until is not None
            sel = node_selector(until)
            res = f"{CmdNames.RUN_UNTIL} {sel}"
            if hs:
                res += f" {hints(hs)}"
            return res
        case demos.SelectSpace(ref):
            if ts.expects_query:
                return f"{CmdNames.ANSWER} {space_ref(ref)}"
            else:
                return f"{CmdNames.SELECT} {space_ref(ref)}"
        case demos.GoToChild(action):
            return f"{CmdNames.GO_TO_CHILD} {value_ref(action)}"
        case demos.IsSuccess():
            return CmdNames.IS_SUCCESS
        case demos.IsFailure():
            return CmdNames.IS_FAILURE
        case demos.Save(name):
            return f"{CmdNames.SAVE} {name}"
        case demos.Load(name):
            return f"{CmdNames.LOAD} {name}"


def test_command(tc: demos.TestCommand) -> str:
    return " | ".join(test_step(ts) for ts in tc)
