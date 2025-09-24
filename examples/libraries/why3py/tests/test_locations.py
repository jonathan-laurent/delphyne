import pytest

import why3py
from why3py.locations import HighlightHint, highlight, split_error_location


def test_highlight():
    s = "ABC\nDEF"
    hs = [
        HighlightHint((1, 0, 1, 1), "(", ")"),
        HighlightHint((1, 1, 1, 2), "[", "]"),
        HighlightHint((1, 2, 2, 1), "{", "}"),
    ]
    assert highlight(s, hs) == "(A)[B]{C\nD}EF"


TYPE_ERROR_ONE_LINE = """
        use int.Int

        let main () diverges =
        let x = (
            1 +
            "hello") in
        x
    """

TYPE_ERROR_TWO_LINES = """
    use int.Int

    let main () diverges =
    let x: int = (any
        string)
    in
    x
    """


@pytest.mark.parametrize("src", [TYPE_ERROR_ONE_LINE, TYPE_ERROR_TWO_LINES])
def test_split_error_location(src):
    match why3py.prove(src, max_steps=5000, max_time_in_secs=5.0):
        case ("Error", (err,)):
            assert split_error_location(err) is not None
        case _:
            assert False
