"""
Parser for references and test commands

See tests/test_parser for examples.
"""

# pyright: basic
# ruff: noqa: E731  # no lambda definition

from collections.abc import Sequence
from typing import cast

import parsy as ps
import yaml  # pyright: ignore[reportMissingTypeStubs]

from delphyne.core import demos, hrefs, irefs, refs
from delphyne.core.demos import CmdNames
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


def _hint_based_atomic_value_ref_from_index_list(
    ref: hrefs.SpaceElementRef, indexes: Sequence[int]
) -> hrefs.AtomicValueRef:
    if not indexes:
        return ref
    else:
        return hrefs.IndexedRef(
            _hint_based_atomic_value_ref_from_index_list(ref, indexes[:-1]),
            indexes[-1],
        )


def _id_based_atomic_value_ref_from_index_list(
    ref: irefs.SpaceElementRef, indexes: Sequence[int]
) -> irefs.AtomicValueRef:
    if not indexes:
        return ref
    else:
        return irefs.IndexedRef(
            _id_based_atomic_value_ref_from_index_list(ref, indexes[:-1]),
            indexes[-1],
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
_node_id = _s("%") >> _num.map(irefs.NodeId)
_answer_id = _s("@") >> _num.map(irefs.AnswerId)
_space_id = _s("$") >> _num.map(irefs.SpaceId)

# Hint
_hint_qual = _ident
_hint_qual_colon = _hint_qual << _s(":")
_hint_val = ps.regex("#?" + _ident_regex)
_hint = ps.seq(_hint_qual_colon.optional(), _hint_val).combine(hrefs.Hint)
_hints = _s("'") >> _hint.sep_by(_space).map(tuple) << _s("'")

# SpaceName
_sname_index = _s("[") >> _num << _s("]")
_sname_indexes = _sname_index.at_least(1).map(tuple)
_sname = ps.seq(_ident, _sname_indexes.optional(())).combine(refs.SpaceName)


# Hint-based SpaceRef
_hvref = ps.forward_declaration()
_hsargs = _s("(") >> _hvref.sep_by(_comma) << _s(")")
_hsref = ps.seq(_sname, _hsargs.optional(()).map(tuple)).combine(
    hrefs.SpaceRef
)

# Hint-based SpaceElementRef
_hseref_hints = _hints.map(lambda hs: hrefs.SpaceElementRef(None, hs))
_hseref_long = ps.seq(_hsref, _s("{") >> _hints << _s("}")).combine(
    hrefs.SpaceElementRef
)
_hseref = _hseref_hints | _hseref_long

# Hint-based ValueRef
_havref = ps.seq(_hseref, (_s("[") >> _num << _s("]")).many()).combine(
    _hint_based_atomic_value_ref_from_index_list
)
_hnoneref = _s(refs.NONE_REF_REPR).map(lambda _: None)
_hvref.become(_hnoneref | _tuple(_hvref) | _havref)


# Id-based SpaceRef
_ivref = ps.forward_declaration()
_isargs = _s("(") >> _ivref.sep_by(_comma) << _s(")")
_isref = ps.seq(_sname, _isargs.optional(()).map(tuple)).combine(
    irefs.SpaceRef
)

# Id-based SpaceElementRef
_iseref = ps.seq(
    _space_id, _s("{") >> (_node_id | _answer_id) << _s("}")
).combine(irefs.SpaceElementRef)

# Id-based ValueRef
_iavref = ps.seq(_iseref, (_s("[") >> _num << _s("]")).many()).combine(
    _id_based_atomic_value_ref_from_index_list
)
_inoneref = _s(refs.NONE_REF_REPR).map(lambda _: None)
_ivref.become(_inoneref | _tuple(_ivref) | _iavref)


# NodeOrigin, SpaceOrigin
_child = ps.seq(_node_id, _comma >> _ivref).combine(irefs.ChildOf)
_child = _s("child(") >> _child << _s(")")
_nested = (_s("nested(") >> _space_id << _s(")")).map(irefs.NestedIn)
_node_origin = _child | _nested

_space_origin = ps.seq(_node_id, _s(".") >> _isref).combine(irefs.SpaceOrigin)

# Node selectors
_tag_sel = ps.seq(_ident, (_s("#") >> _num).optional())
_tag_sel = _tag_sel.combine(demos.TagSelector)
_tag_sels = _tag_sel.sep_by(_s("&"), min=1)
_node_sel = _tag_sels.sep_by(_s("/"), min=1).map(_node_selector_from_path)

# Test commands
_run = (
    _s(CmdNames.RUN)
    >> _spopt
    >> _hints.optional(default=()).map(lambda hs: demos.Run(hs, None))
)
_until = (
    _s(CmdNames.RUN_UNTIL)
    >> _space
    >> ps.seq(_node_sel, (_space >> _hints).optional(default=())).combine(
        lambda u, hs: demos.Run(hs, u)
    )
)
_gosub = (_s(CmdNames.SELECT) >> _space >> _hsref).map(
    lambda ref: demos.SelectSpace(ref, expects_query=False)
)
_go_child = (_s(CmdNames.GO_TO_CHILD) >> _space >> _hvref).map(demos.GoToChild)
_answer = (_s(CmdNames.ANSWER) >> _space >> _hsref).map(
    lambda ref: demos.SelectSpace(ref, expects_query=True)
)
_success = _s(CmdNames.IS_SUCCESS).map(lambda _: demos.IsSuccess())
_failure = _s(CmdNames.IS_FAILURE).map(lambda _: demos.IsFailure())
_save = _s(CmdNames.SAVE) >> _space >> _ident.map(demos.Save)
_load = _s(CmdNames.LOAD) >> _space >> _ident.map(demos.Load)
_tstep = (
    _run
    | _until
    | _gosub
    | _go_child
    | _answer
    | _success
    | _failure
    | _save
    | _load
)
_test = _spopt >> _tstep.sep_by(_spopt >> _s("|") << _spopt) << _spopt


#####
##### External Interface
#####


ParseError = ps.ParseError


def hint_based_value_ref(s: str) -> hrefs.ValueRef:
    return cast(hrefs.ValueRef, _hvref.parse(s))


def hint_based_space_ref(s: str) -> hrefs.SpaceRef:
    return cast(hrefs.SpaceRef, _hsref.parse(s))


def id_based_value_ref(s: str) -> irefs.ValueRef:
    return cast(irefs.ValueRef, _ivref.parse(s))


def id_based_space_ref(s: str) -> irefs.SpaceRef:
    return cast(irefs.SpaceRef, _isref.parse(s))


def node_origin(s: str) -> irefs.NodeOrigin:
    return cast(irefs.NodeOrigin, _node_origin.parse(s))


def space_origin(s: str) -> irefs.SpaceOrigin:
    return cast(irefs.SpaceOrigin, _space_origin.parse(s))


def test_command(s: str) -> demos.TestCommand:
    return cast(demos.TestCommand, _test.parse(s))


def demo_file(input: str) -> demos.DemoFile:
    return pydantic_load(demos.DemoFile, yaml.safe_load(input))
