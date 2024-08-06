import timeit

import utils

from why3py.core import prove
from why3py.locations import highlight_mlw
from why3py.utils import print_rich


TEST_MLW = """
use int.Int

let loop ()
    diverges
    ensures { result >= 0 }
=
    let ref x = 0 in
    while x < 10 do
        invariant x_nonneg { x >= 0 }
        if x = 5 then
            x <- x - 1
        else
            x <- x + 1
    done;
    x
"""


def test_prove_obligations(monkeypatch):
    utils.force_atty(monkeypatch)
    t0 = timeit.default_timer()
    res = prove(TEST_MLW)
    t1 = timeit.default_timer()
    print(f"Time to return: {t1 - t0:.2f} s", end="\n\n")
    match res:
        case ("Answer", (obls,)):
            for obl in obls:
                print_rich(
                    f"[bold yellow]Obligation {obl['name']}:[/bold yellow]",
                    end="\n",
                )
                print_rich(highlight_mlw(TEST_MLW, obl["locs"]))
        case _:
            assert False
