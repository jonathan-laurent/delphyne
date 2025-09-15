import pytest

from delphyne.core import parse, pprint


@pytest.mark.parametrize(
    "s",
    [
        "foo",
        "sub[1]",
        "aggr(['', 'foo bar'])",
        "cex(cands{'foo'})",
        "foo(sub[1]{%2}, cands{@3})",
        "next(nil)",
        "next(next(nil){''}[1])",
    ],
)
def test_space_ref_roundabout(s: str):
    parsed = parse.space_ref(s)
    hash(parsed)  # should be hashable
    assert pprint.space_ref(parsed) == s


@pytest.mark.parametrize("s", ["nested(1, gen)", "child(1, gen{@3})"])
def test_node_origin_roundabout(s: str):
    parsed = parse.node_origin(s)
    hash(parsed)
    assert pprint.node_origin(parsed) == s


@pytest.mark.parametrize(
    "inp,out",
    [
        ("run", None),
        ("run | success", None),
        ("run | failure", None),
        ("run 'foo baz:qux'", None),
        ("run 'foo #bar'", None),  # value hint
        ("at find_inv", None),
        ("at find_inv 'foo bar'", None),
        ("go aggr(['', 'foo bar'])", None),
        ("at find_inv | go sub[1]", None),
        ("at find_inv | answer aggr(['', 'alt'])", None),
        ("at EvalProg | take nil", None),
        ("at join_node | take ['', 'foo']", None),
        ("at my_branch | take cands{'foo'}", None),
        (" run  \n | run", "run | run"),
        ("save x | load x", None),
        # Advanced node selectors
        ("at tag#2", None),
        ("at tag#1/iter/node#3", None),
        ("at tag1&tag2/iter", None),
        ("at tag1#1&tag2#2/iter", None),
    ],
)
def test_parser(inp: str, out: str | None):
    parsed = parse.test_command(inp)
    printed = pprint.test_command(parsed)
    if out:
        assert printed == out
    else:
        assert printed == inp
    print(parsed)
    print(pprint.test_command(parsed))
