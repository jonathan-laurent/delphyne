import heapq
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import cast

from why3py.core import Color, Loc, Update


@dataclass
class HighlightHint:
    loc: Loc
    before: str
    after: str


def _loc_start(l: Loc) -> tuple[int, int]:
    bl, bc, _, _ = l
    return (bl, bc)


def _loc_end(l: Loc) -> tuple[int, int]:
    _, _, el, ec = l
    return (el, ec)


@dataclass(order=True)
class _HintWithPos:
    pos: tuple[int, int]
    hint: HighlightHint = field(compare=False)


def highlight(s: str, hs: list[HighlightHint]) -> str:
    start = [_HintWithPos(_loc_start(h.loc), h) for h in hs]
    heapq.heapify(start)
    end = []
    res = ""
    # current line and column in the source string (just before 'char')
    l, c = (1, 0)
    for char in s:
        if char == "\n":
            l += 1
            c = 0
            res += "\n"
            continue
        while start and (l, c) == start[0].pos:
            h = heapq.heappop(start).hint
            res += h.before
            heapq.heappush(end, _HintWithPos(_loc_end(h.loc), h))
        res += char
        c += 1
        while end and (l, c) == end[0].pos:
            h = heapq.heappop(end).hint
            res += h.after
    return res


def highlight_mlw(src: str, locs: list[tuple[Loc, Color]]) -> str:
    hs = []
    for l, c in locs:
        m = "magenta" if c[0] == "Goal" else "cyan"
        hs.append(HighlightHint(l, f"[{m}]", f"[/{m}]"))
    return highlight(src, hs)


def highlight_diff(edited: str, diffs: list[Update]) -> str:
    hs = []
    for d in diffs:
        match d:
            case ("Added_invariant", (loc,)):
                hs.append(HighlightHint(loc, "[bold green]", "[/bold green]"))
    return highlight(edited, hs)


def comment_locations(src: str, comments: list[tuple[Loc, str]]) -> str:
    lines = src.splitlines()
    added = defaultdict(set)
    for loc, comment in comments:
        bl, _, el, _ = loc
        for l in range(bl, el + 1):
            added[l].add(comment)
    for l, to_add in added.items():
        lines[l - 1] += "  (* " + ", ".join(to_add) + " *)"
    return "\n".join(lines)


def comment_of_color(c: Color) -> str:
    match c[0]:
        case "Goal":
            return "GOAL"
        case "Premise":
            return "premise"


def annotate_premises_and_goals(
    src: str, colors: list[tuple[Loc, Color]]
) -> str:
    hints = [(loc, comment_of_color(kind)) for loc, kind in colors]
    return comment_locations(src, hints)


def loc_neighborhood(src: str, loc: Loc, radius=1, add_comments=True) -> str:
    lines = src.splitlines()
    bl, _, el, _ = loc
    if add_comments:
        if bl == el:
            lines[bl - 1] += "  (* ERROR *)"
        else:
            lines[bl - 1] += "  (* START ERROR *)"
            lines[el - 1] += "  (* END ERROR *)"
    start = max(1, bl - radius) - 1
    end = min(len(lines), el + radius) - 1
    return "\n".join(lines[start : end + 1])


@dataclass
class LocatedError:
    msg: str
    loc: Loc


def split_error_location(msg: str) -> LocatedError | None:
    pat = r"File line (\d+), characters (\d+)-(\d+):(.*)"
    match = re.search(pat, msg, re.MULTILINE | re.DOTALL)
    if match is not None:
        loc = tuple(int(match.group(i)) for i in [1, 2, 1, 3])
        loc = cast(tuple[int, int, int, int], loc)
        return LocatedError(match.group(4).strip(), loc)
    pat = (
        r"File line (\d+), character (\d+) to line (\d+), character (\d+):(.*)"
    )
    match = re.search(pat, msg, re.MULTILINE | re.DOTALL)
    if match is not None:
        loc = tuple(int(match.group(i)) for i in [1, 2, 3, 4])
        loc = cast(tuple[int, int, int, int], loc)
        return LocatedError(match.group(5).strip(), loc)
    return None
