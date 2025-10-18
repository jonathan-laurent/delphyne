from collections.abc import Sequence

import pytest


@pytest.mark.parametrize(
    "pattern,spaces,label,expected",
    [
        ("foo", [], "foo", True),
        ("foo", [], "bar", False),
        ("foo", [["space"]], "foo", False),
        ("1/2/foo", [["space", "1"], ["space", "2"]], "foo", True),
        ("1/3/foo", [["space", "1"], ["space", "2"]], "foo", False),
        (
            "space&1/space&2/foo",
            [["space", "1"], ["space", "2"]],
            "foo",
            True,
        ),
    ],
)
def test_match_handler_pattern(
    pattern: str,
    spaces: Sequence[Sequence[str]],
    label: str,
    expected: bool,
):
    from delphyne.stdlib.feedback_processing import match_handler_pattern

    assert match_handler_pattern(pattern, spaces, label) == expected
