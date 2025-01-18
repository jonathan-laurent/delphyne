"""
A very basic API to interact with Why3.
"""

from collections.abc import Sequence
from dataclasses import dataclass

type File = str
type Formula = str


@dataclass
class Obligation:
    name: str
    proved: bool
    relevance_hints: str


@dataclass
class Feedback:
    error: str | None
    obligations: Sequence[Obligation]

    @property
    def success(self) -> bool:
        return self.error is None and all(obl.proved for obl in self.obligations)


def check(prog: File, annotated: File) -> Feedback:
    """
    Verify the correctness of a WhyML program.

    This function takes as an input a program `prog`, along with a copy
    of this same program with additional proof annotations (in the form
    of additional assertions or loop invariants).

    It returns a feedback object with two fields:

    - `error`: if the program is invalid or the annotated version is not
      consistent with the original program, this field is a string
      explaining the error. Otherwise, it is none.
    - `obligations`: a list of proof obligations (empty if an error
      occured). For each obligation, we list its name (as shown in the
      Why3 tree view), whether or not it was successfully proved, and
      relevance hints in the form of a version of the annotated program
      with additional comments, indicating at each line whether it is
      relevant to either a premise or  conclusion of the proof
      obligation. This information is displayed in the Why3 IDE using
      colors.
    """
    import why3py.simple as why3py

    outcome = why3py.check_file(annotated, prog)
    if outcome.kind == "error" or outcome.kind == "mismatch":
        return Feedback(error=why3py.summary(outcome), obligations=[])
    obligations = [
        Obligation(obl.name, obl.proved, obl.annotated) for obl in outcome.obligations
    ]
    return Feedback(error=None, obligations=obligations)


def add_invariant(prog: File, inv: Formula) -> File:
    """
    Add an invariant to a single-loop program.

    TODO: make this function more robust.
    """
    lines = prog.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("  while"):
            lines.insert(i + 1, f"    invariant {{ {inv} }}")
            return "\n".join(lines)
    assert False


def add_invariants(prog: File, invs: Sequence[Formula]) -> File:
    for inv in reversed(invs):
        prog = add_invariant(prog, inv)
    return prog
