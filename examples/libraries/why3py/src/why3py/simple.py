"""
A clean, self-contained, minimalistic interface to Why3.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Literal, cast

import why3py

type File = str


type Outcome = Obligations | Mismatch | Error


type Color = Literal["premise", "goal"]


def _color_of_why3py(col: why3py.Color) -> Color:
    if col[0] == "Premise":
        return "premise"
    elif col[0] == "Goal":
        return "goal"


@dataclass
class Obligation:
    name: str  # name, as indicated in the menu
    context: str  # premises of the proof obligation sequent
    goal: str  # conclusion of the proof obligation sequent
    goal_loc: why3py.Loc | None
    locs: list[tuple[why3py.Loc, Color]]  # file highlighting
    annotated: str  # annotated original file
    proved: bool
    prover_answer: str
    prover_steps: int  # number of steps taken by the prover

    @staticmethod
    def of_why3py(file: File, obl: why3py.Obligation) -> "Obligation":
        obligation = why3py.clean_identifiers(obl["task"])
        split_obligation = why3py.split_sequent(obligation)
        return Obligation(
            name=obl["name"],
            context=split_obligation.context,
            goal=split_obligation.goal,
            goal_loc=obl["goal_loc"],
            locs=[(l, _color_of_why3py(c)) for (l, c) in obl["locs"]],
            annotated=why3py.annotate_premises_and_goals(file, obl["locs"]),
            proved=obl["proved"],
            prover_answer=obl["prover_answer"],
            prover_steps=obl["prover_steps"],
        )


@dataclass
class Obligations:
    kind: Literal["obligations"]
    obligations: list[Obligation]

    @staticmethod
    def of_why3py(file: File, obls: list[why3py.Obligation]) -> "Obligations":
        return Obligations(
            kind="obligations",
            obligations=[Obligation.of_why3py(file, obl) for obl in obls],
        )

    @property
    def success(self):
        return all(obl.proved for obl in self.obligations)


@dataclass
class Mismatch:
    kind: Literal["mismatch"]
    loc: tuple[why3py.Loc | None]


@dataclass
class Error:
    kind: Literal["error"]
    message: str


def check_file(
    file: File,
    original: File | None,
    *,
    max_steps: int | None = None,
    max_time_in_seconds: float | None = None,
) -> Outcome:
    if max_steps is None:
        max_steps = why3py.DEFAULT_MAX_STEPS
    if max_time_in_seconds is None:
        max_time_in_seconds = why3py.DEFAULT_MAX_TIME_IN_SECONDS
    try:
        if original is not None:
            diff = why3py.answer(why3py.diff(original, file))
            if diff[0] == "Mismatch":
                return Mismatch(kind="mismatch", loc=diff[1])
        obligations = why3py.answer(
            why3py.prove(
                file,
                max_steps=max_steps,
                max_time_in_secs=max_time_in_seconds,
            )
        )
        return Obligations.of_why3py(file, obligations)
    except why3py.Why3Error as e:
        return Error(kind="error", message=e.msg)
    except Exception:
        import sys

        e = sys.exc_info()
        return Error(kind="error", message=f"Unknown Why3 exception: {e}")


def summary(result: Outcome) -> str:
    match result:
        case Obligations():
            if result.success:
                return "Success"
            else:
                obls = [o.name for o in result.obligations if not o.proved]
                return "Some obligations remain: " + ", ".join(obls)
        case Mismatch():
            return "Mismatch detected"
        case Error():
            return result.message


#####
##### Analyzing annotations
#####


def lines_of_unproved_goals(obls: Obligations) -> Iterable[int]:
    for obl in obls.obligations:
        if obl.proved:
            continue
        if obl.goal_loc is not None:
            for i in range(obl.goal_loc[0], obl.goal_loc[2] + 1):
                yield i


type AnnotKind = Literal["assert", "invariant"]


@dataclass(frozen=True)
class Annot:
    line: int
    kind: AnnotKind
    formula: str


def parse_annotation_line(s: str) -> tuple[AnnotKind, str] | None:
    """
    From a line starting with `assert {...}` or `invariant {...}`,
    return an `Annot`. The line can start with an arbitrary number of
    whitespaces.
    """
    import re

    match = re.match(r"\s*(assert|invariant)\s*{(.*)}\s*", s)
    if match is None:
        return None
    return cast(Any, (match.group(1), match.group(2).strip()))


def unproved_lines(prog: File, obls: Obligations) -> list[str]:
    lines = set(lines_of_unproved_goals(obls))
    prog_lines = prog.splitlines()
    return [prog_lines[i - 1] for i in lines]


def unproved_annotations(prog: File, obls: Obligations) -> Iterable[Annot]:
    lines = prog.splitlines()
    for i in lines_of_unproved_goals(obls):
        line = lines[i - 1]
        if (p := parse_annotation_line(line)) is not None:
            yield Annot(i, p[0], p[1])


def all_annotations(prog: File) -> Iterable[Annot]:
    for i, l in enumerate(prog.splitlines()):
        if (p := parse_annotation_line(l)) is not None:
            yield Annot(i + 1, p[0], p[1])


type AnnotStatus = Literal["proved", "unproved"]


def annotations_with_status(
    prog: File, obls: Obligations
) -> dict[Annot, AnnotStatus]:
    res: dict[Annot, AnnotStatus] = {
        a: "proved" for a in all_annotations(prog)
    }
    for a in unproved_annotations(prog, obls):
        res[a] = "unproved"
    return res
