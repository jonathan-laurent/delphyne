"""
Parser for references and test commands
"""

# pyright: basic
# ruff: noqa: E731  # no lambda definition

import parsy as ps
import yaml  # pyright: ignore[reportMissingTypeStubs]

from delphyne.core import demos, refs
from delphyne.core.pprint import CmdNames
from delphyne.utils.typing import pydantic_load

"""
Grammar definition with Parsy
"""


# Allowed identifiers
_ident = ps.regex(r"[a-zA-Z_][a-zA-Z0-9\-\._]*")

# Utilities
_s = ps.string
_comma = _s(",") >> ps.whitespace.many()
_spopt = ps.whitespace.many()
_space = ps.whitespace.at_least(1)
_num = ps.digit.at_least(1).concat().map(int)
_tuple = lambda p: (_s("[") >> p.sep_by(_comma) << _s("]")).map(tuple)

# NodeId | AnswerId
_node_id = _s("%") >> _num.map(refs.NodeId)
_answer_id = _s("@") >> _num.map(refs.AnswerId)

# Hint
_hintsel = _ident
_hintsel_colon = _hintsel << _s(":")
_hint = ps.seq(_hintsel_colon.optional(), _ident).combine(refs.Hint)
_hints = _s("'") >> _hint.sep_by(_space).map(tuple) << _s("'")
_hints_ref = _hints.map(refs.HintsRef)

# SpaceName
_sname_index = _s("[") >> _num << _s("]")
_sname_indexes = _sname_index.at_least(1).map(tuple)
_sname = ps.seq(_ident, _sname_indexes.optional(())).combine(refs.SpaceName)

# SpaceRef
_vref = ps.forward_declaration()
_sargs = _s("(") >> _vref.sep_by(_comma) << _s(")")
_sref = ps.seq(_sname, _sargs.optional(()).map(tuple)).combine(refs.SpaceRef)

# SpaceElementRef
_seref_hints_val = _hints_ref
_seref_val = _node_id | _answer_id | _seref_hints_val
_seref_hints = _seref_hints_val.map(lambda hs: refs.SpaceElementRef(None, hs))
_seref_long = ps.seq(_sref, _s("{") >> _seref_val << _s("}"))
_seref_long = _seref_long.combine(refs.SpaceElementRef)
_seref = _seref_hints | _seref_long

# ValueRef
_vref.become(_tuple(_vref) | _seref)

# NodeOrigin
_naked_nid = _num.map(refs.NodeId)
_child = ps.seq(_naked_nid, _comma >> _vref).combine(refs.ChildOf)
_child = _s("child(") >> _child << _s(")")
_nested = ps.seq(_naked_nid, _comma >> _sref).combine(refs.NestedTreeOf)
_nested = _s("nested(") >> _nested << _s(")")
_node_origin = _child | _nested

# Test commands
_run = _s(CmdNames.RUN) >> _spopt >> _hints.optional(default=())
_run = _run.map(lambda hs: demos.Run(hs, None))
_until = ps.seq(_hintsel, (_space >> _hints).optional(default=()))
_until = _until.combine(lambda u, hs: demos.Run(hs, u))
_until = _s(CmdNames.RUN_UNTIL) >> _space >> _until
_gosub = _s(CmdNames.SUBCHOICE) >> _space >> _sref
_gosub = _gosub.map(lambda ref: demos.SelectSpace(ref, expects_query=False))
_answer = _s(CmdNames.ANSWER) >> _space >> _sref
_answer = _answer.map(lambda ref: demos.SelectSpace(ref, expects_query=True))
_success = _s(CmdNames.IS_SUCCESS).map(lambda _: demos.IsSuccess())
_failure = _s(CmdNames.IS_FAILURE).map(lambda _: demos.IsFailure())
_save = _s(CmdNames.SAVE) >> _space >> _ident.map(demos.Save)
_load = _s(CmdNames.LOAD) >> _space >> _ident.map(demos.Load)
_tstep = _run | _until | _gosub | _answer | _success | _failure | _save | _load
_test = _spopt >> _tstep.sep_by(_spopt >> _s("|") << _spopt) << _spopt


"""
External interface
"""


ParseError = ps.ParseError


def value_ref(s: str) -> refs.ValueRef:
    return _vref.parse(s)


def space_ref(s: str) -> refs.SpaceRef:
    return _sref.parse(s)


def node_origin(s: str) -> refs.NodeOrigin:
    return _node_origin.parse(s)


def test_command(s: str) -> demos.TestCommand:
    return _test.parse(s)


def demo_file(input: str) -> demos.DemoFile:
    return pydantic_load(demos.DemoFile, yaml.safe_load(input))
