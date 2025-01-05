"""
Pretty-printing references.

Paths and inline answers are not supported and so exceptions are raised
when these occur.
"""

from collections.abc import Callable, Sequence
from typing import cast

from delphyne.core import demos, refs


class PathDetected(Exception): ...


class AnswerDetected(Exception): ...


def node_id(id: refs.NodeId) -> str:
    return f"%{id.id}"


def answer_id(id: refs.AnswerId) -> str:
    return f"@{id.id}"


def assembly[T](pp: Callable[[T], str], a: refs.Assembly[T]) -> str:
    """Print an assembly, assuming that T does not intersect with tuple."""
    if isinstance(a, tuple):
        a = cast(tuple[refs.Assembly[T], ...], a)
        return "[" + ", ".join(assembly(pp, x) for x in a) + "]"
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


def value_ref(vr: refs.ValueRef) -> str:
    return assembly(space_element_ref, vr)


def hint(h: refs.Hint) -> str:
    if not h.query_name:
        return h.hint
    return f"{h.query_name}:{h.hint}"


def hints(hs: Sequence[refs.Hint]):
    return "'" + " ".join(hint(h) for h in hs) + "'"


def space_element_ref(ser: refs.SpaceElementRef) -> str:
    if ser.space is None:
        assert isinstance(ser.element, refs.Hints)
        return hints(ser.element.hints)
    match ser.element:
        case refs.Answer():
            raise AnswerDetected()
        case refs.AnswerId():
            value = answer_id(ser.element)
        case refs.NodeId():
            value = node_id(ser.element)
        case refs.Hints():
            value = hints(ser.element.hints)
        case _:
            assert isinstance(ser.element, tuple)
            raise PathDetected()
    return f"{space_ref(ser.space)}{{{value}}}"


def node_origin(no: refs.NodeOrigin) -> str:
    match no:
        case refs.ChildOf(node, action):
            return f"child({node.id}, {value_ref(action)})"
        case refs.NestedTreeOf(node, choice):
            return f"nested({node.id}, {space_ref(choice)})"


class CmdNames:
    RUN = "run"
    RUN_UNTIL = "at"
    SUBCHOICE = "go"
    ANSWER = "answer"
    IS_SUCCESS = "success"
    IS_FAILURE = "failure"
    SAVE = "save"
    LOAD = "load"


def test_step(ts: demos.TestStep) -> str:
    match ts:
        case demos.Run(hs, None):
            if not hs:
                return CmdNames.RUN
            return f"{CmdNames.RUN} {hints(hs.hints)}"
        case demos.Run(hs, until):
            assert until is not None
            res = f"{CmdNames.RUN_UNTIL} {until}"
            if hs:
                res += f" {hints(hs.hints)}"
            return res
        case demos.SelectSpace(ref):
            if ts.expects_query:
                return f"{CmdNames.ANSWER} {space_ref(ref)}"
            else:
                return f"{CmdNames.SUBCHOICE} {space_ref(ref)}"
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
