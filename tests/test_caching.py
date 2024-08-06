from pathlib import Path
from shutil import rmtree

from delphyne.stdlib.caching import cache


CACHE_DIR = Path(__file__).parent / "cache" / "sum_list_str"


@cache(CACHE_DIR)
def sum_list_str(xs: list[int]) -> str:
    return str(sum(xs))


def test_cache():
    rmtree(CACHE_DIR, ignore_errors=True)
    inps = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]]
    outs1 = [sum_list_str(inp) for inp in inps]
    outs2 = [sum_list_str(inp) for inp in inps]
    rmtree(CACHE_DIR, ignore_errors=True)
    assert outs1 == outs2


test_cache()
