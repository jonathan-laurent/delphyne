"""
Parser for references and test commands

See tests/test_parser for examples.
"""

# pyright: basic
# ruff: noqa: E731  # no lambda definition

from collections.abc import Sequence

import parsy as ps
import yaml  # pyright: ignore[reportMissingTypeStubs]

from delphyne.core import demos, refs
from delphyne.core.pprint import NONE_REF_REPR, CmdNames
from delphyne.utils.typing import pydantic_load

#####
##### Utilities
#####


def _node_selector_from_path(
    sels: list[demos.TagSelectors],
) -> demos.NodeSelector:
    if len(sels) == 1:
        return sels[0]
    else:
        return demos.WithinSpace(sels[0], _node_selector_from_path(sels[1:]))


def _atomic_value_ref_from_index_list(
    ref: refs.SpaceElementRef, indexes: Sequence[int]
) -> refs.AtomicValueRef:
    if not indexes:
        return ref
    else:
        return refs.IndexedRef(
            _atomic_value_ref_from_index_list(ref, indexes[:-1]), indexes[-1]
        )


#####
##### Grammar Definition with Parsy
#####


# Allowed identifiers
_ident_regex = r"[a-zA-Z_][a-zA-Z0-9\-\._]*"
_ident = ps.regex(_ident_regex)

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
_hint_qual = _ident
_hint_qual_colon = _hint_qual << _s(":")
_hint_val = ps.regex("#?" + _ident_regex)
_hint = ps.seq(_hint_qual_colon.optional(), _hint_val).combine(refs.Hint)
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

# AtomicValueRef
_avref = ps.seq(_seref, (_s("[") >> _num << _s("]")).many())
_avref = _avref.combine(_atomic_value_ref_from_index_list)

# ValueRef
_noneref = _s(NONE_REF_REPR).map(lambda _: None)
_vref.become(_noneref | _tuple(_vref) | _avref)

# NodeOrigin
_naked_nid = _num.map(refs.NodeId)
_child = ps.seq(_naked_nid, _comma >> _vref).combine(refs.ChildOf)
_child = _s("child(") >> _child << _s(")")
_nested = ps.seq(_naked_nid, _comma >> _sref).combine(refs.NestedTreeOf)
_nested = _s("nested(") >> _nested << _s(")")
_node_origin = _child | _nested


# Node selectors
_tag_sel = ps.seq(_ident, (_s("#") >> _num).optional())
_tag_sel = _tag_sel.combine(demos.TagSelector)
_tag_sels = _tag_sel.sep_by(_s("&"), min=1)
_node_sel = _tag_sels.sep_by(_s("/"), min=1).map(_node_selector_from_path)


# Test commands
_run = _s(CmdNames.RUN) >> _spopt >> _hints.optional(default=())
_run = _run.map(lambda hs: demos.Run(hs, None))
_until = ps.seq(_node_sel, (_space >> _hints).optional(default=()))
_until = _until.combine(lambda u, hs: demos.Run(hs, u))
_until = _s(CmdNames.RUN_UNTIL) >> _space >> _until
_gosub = _s(CmdNames.SELECT) >> _space >> _sref
_gosub = _gosub.map(lambda ref: demos.SelectSpace(ref, expects_query=False))
_go_child = _s(CmdNames.GO_TO_CHILD) >> _space >> _vref
_go_child = _go_child.map(demos.GoToChild)
_answer = _s(CmdNames.ANSWER) >> _space >> _sref
_answer = _answer.map(lambda ref: demos.SelectSpace(ref, expects_query=True))
_success = _s(CmdNames.IS_SUCCESS).map(lambda _: demos.IsSuccess())
_failure = _s(CmdNames.IS_FAILURE).map(lambda _: demos.IsFailure())
_save = _s(CmdNames.SAVE) >> _space >> _ident.map(demos.Save)
_load = _s(CmdNames.LOAD) >> _space >> _ident.map(demos.Load)
_tstep = _run | _until | _gosub | _go_child | _answer | _success | _failure
_tstep = _tstep | _save | _load
_test = _spopt >> _tstep.sep_by(_spopt >> _s("|") << _spopt) << _spopt


#####
##### External Interface
#####


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
