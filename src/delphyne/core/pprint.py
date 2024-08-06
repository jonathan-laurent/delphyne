"""
Pretty-printing references.
"""

from collections.abc import Callable, Sequence
from typing import cast

from delphyne.core import demos, refs


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


def choice_ref(cr: refs.ChoiceRef) -> str:
    label, args = cr
    if not args:
        return label
    args_str = ", ".join(assembly(choice_arg, a) for a in args)
    return f"{label}({args_str})"


def choice_arg(ca: refs.ChoiceArgRef) -> str:
    return str(ca) if isinstance(ca, int) else value_ref(ca)


def value_ref(vr: refs.ValueRef) -> str:
    return assembly(choice_outcome_ref, vr)


def hint(h: refs.Hint) -> str:
    if not h.query_name:
        return h.hint
    return f"{h.query_name}:{h.hint}"


def hints(hs: Sequence[refs.Hint]):
    return "'" + " ".join(hint(h) for h in hs) + "'"


def choice_outcome_ref(cor: refs.ChoiceOutcomeRef) -> str:
    if cor.choice is None:
        assert isinstance(cor.value, refs.Hints)
        return hints(cor.value.hints)
    match cor.value:
        case refs.AnswerId():
            value = answer_id(cor.value)
        case refs.NodeId():
            value = node_id(cor.value)
        case refs.Hints():
            value = hints(cor.value.hints)
    return f"{choice_ref(cor.choice)}{{{value}}}"


def node_origin(no: refs.NodeOrigin) -> str:
    match no:
        case refs.ChildOf(node, action):
            return f"child({node.id}, {value_ref(action)})"
        case refs.SubtreeOf(node, choice):
            return f"subtree({node.id}, {choice_ref(choice)})"


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
            return f"{CmdNames.RUN} {hints(hs)}"
        case demos.Run(hs, until):
            assert until is not None
            res = f"{CmdNames.RUN_UNTIL} {until}"
            if hs:
                res += f" {hints(hs)}"
            return res
        case demos.SelectSub(ref):
            if ts.expects_query:
                return f"{CmdNames.ANSWER} {choice_ref(ref)}"
            else:
                return f"{CmdNames.SUBCHOICE} {choice_ref(ref)}"
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
