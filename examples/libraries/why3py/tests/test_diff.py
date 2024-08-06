import utils

from why3py.core import diff
from why3py.locations import highlight_diff
from why3py.utils import print_rich


def test_diff(monkeypatch):
    loop = """
    let loop () diverges =
        let ref x = 0 in
        while x < 10 do
            x <- x + 1;
            while any bool do
                x <- x + 2
            done
        done;
        x
    """
    loop_new = """
    let loop () diverges =
        let ref x = 0 in
        while x < 10 do
            invariant { x >= 0 }
            x <- x + 1;
            while any bool do
                invariant { x >= 1 }
                x <- x + 2
            done
        done;
        x
    """
    utils.force_atty(monkeypatch)
    match diff(loop, loop_new):
        case ("Answer", (("Updates", (updates,)),)):
            print_rich(highlight_diff(loop_new, updates))
        case _:
            assert False
