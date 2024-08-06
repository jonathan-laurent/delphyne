from pathlib import Path

import pytest

import why3py.simple as why3


EXAMPLES = Path(__file__).parent / "examples"


def test_check():
    example = EXAMPLES / "triple_inv_solved.mlw"
    res = why3.check_file(example.read_text(), None)
    assert res.kind == "obligations" and res.success


@pytest.mark.parametrize(
    "ex,unproved,nproved",
    [
        ("triple_inv_intermediate.mlw", [("invariant", "x >= y")], 1),
        ("triple_inv_unsolved.mlw", [("assert", "x >= y")], 0),
        ("triple_inv_solved.mlw", [], 4),
        ("triple_inv_wrong.mlw", [("invariant", "y > 1")], 2),
    ],
)
def test_unproved_annotations(
    ex: str, unproved: list[why3.Annot], nproved: int
):
    example = EXAMPLES / ex
    prog = example.read_text()
    res = why3.check_file(prog, None)
    assert isinstance(res, why3.Obligations)
    all = list(why3.all_annotations(prog))
    assert unproved == [
        (a.kind, a.formula) for a in why3.unproved_annotations(prog, res)
    ]
    assert len(all) - len(unproved) == nproved


@pytest.mark.parametrize(
    "inp,outp",
    [
        (
            "  invariant  { x > 0 && y < 0 } (* new invariant *)",
            ("invariant", "x > 0 && y < 0"),
        ),
        (
            "assert {z > 0} ",
            ("assert", "z > 0"),
        ),
        (" foo {x > 0}", None),
    ],
)
def test_parse_annotation_lines(inp: str, outp: why3.Annot | None):
    assert why3.parse_annotation_line(inp) == outp
