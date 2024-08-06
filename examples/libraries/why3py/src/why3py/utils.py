from dataclasses import dataclass
from typing import Iterable, TypeVar

import rich
import rich.console

from why3py.core import Result


def traverse(obj: object) -> Iterable[object]:
    yield obj
    match obj:
        case list() | tuple():
            for e in obj:
                yield from traverse(e)
        case dict():
            for e in obj.values():
                yield from traverse(e)


def remove_locs(obj: object) -> None:
    for sub in traverse(obj):
        for k in ["expr_loc", "id_loc", "pat_loc"]:
            if isinstance(sub, dict) and k in sub:
                del sub[k]


def print_rich(s: str, **kwargs):
    console = rich.console.Console()
    console.print(s, highlight=False, **kwargs)


@dataclass
class Why3Error(Exception):
    msg: str


T = TypeVar("T")


def answer(res: Result[T]) -> T:
    match res:
        case ("Answer", (x,)):
            return x
        case ("Error", (msg,)):
            raise Why3Error(msg)
