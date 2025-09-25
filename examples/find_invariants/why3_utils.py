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
    goal_formula: str


@dataclass
class Feedback:
    error: str | None
    obligations: Sequence[Obligation]

    def __str__(self):
        # TODO: this nicer representation is not used in the tree
        # explorer because of the way `Compute` nodes work.
        if self.error is not None:
            return self.error
        num_obl = len(self.obligations)
        num_proved = sum(obl.proved for obl in self.obligations)
        return f"{num_proved}/{num_obl} obligations proved"

    @property
    def success(self) -> bool:
        return self.error is None and all(
            obl.proved for obl in self.obligations
        )


def check(
    prog: File, annotated: File, timeout: float | None = None
) -> Feedback:
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

    outcome = why3py.check_file(annotated, prog, max_time_in_seconds=timeout)
    if outcome.kind == "error" or outcome.kind == "mismatch":
        return Feedback(error=why3py.summary(outcome), obligations=[])
    obligations = [
        Obligation(
            obl.name, obl.proved, obl.annotated, _goal_formula(obl.goal)
        )
        for obl in outcome.obligations
    ]
    return Feedback(error=None, obligations=obligations)


#####
##### Naive Program Manipulations
#####


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


def no_invalid_formula_symbol(fml: Formula) -> bool:
    return all(c not in fml for c in "/[]{};")


def invariant_init_obligation(obligation: Obligation) -> bool:
    return "init" in obligation.name


def _goal_formula(descr: str) -> str:
    # Turn a goal description such as `goal vc1: <fml>` into `<fml>`.
    return descr.split(":", 1)[1].strip()


def split_final_assertion(prog: File) -> tuple[File, Formula]:
    # Match "assert { <fml> }"" in `prog` and use a regex to replace it
    # by "assert { true }", while also separately returning <fml>.
    # Ensure that there is one match exactly. This is fragile!
    import re

    # Pattern to match "assert { <formula> }" with flexible whitespace
    pattern = r"assert\s*\{\s*([^}]+)\s*\}"

    matches = list(re.finditer(pattern, prog))

    # Ensure there is exactly one match
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one assert statement, found {len(matches)}"
        )

    match = matches[0]
    formula = match.group(1).strip()

    # Replace the assert statement with "assert { true }"
    modified_prog = (
        prog[: match.start()] + "assert { true }" + prog[match.end() :]
    )

    return modified_prog, formula


def restore_final_assertion(prog: File, fml: Formula) -> File:
    # Match "assert { true }" and replaces `true` by `fml`. This acts as
    # the inverse of `split_final_assertion`.
    import re

    pattern = r"assert\s*\{\s*true\s*\}"
    matches = list(re.finditer(pattern, prog))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one assertion, found {len(matches)}"
        )
    match = matches[0]
    restored_prog = (
        prog[: match.start()] + f"assert {{ {fml} }}" + prog[match.end() :]
    )
    return restored_prog


#####
##### Making Simple Logical Checks
#####


def _get_variable_names(f: Formula) -> Sequence[str]:
    """
    Naive heuristic to extract all variable names from a formula.

    Uses a regex to find all occurences of a valid C-like identifier.
    """

    import re

    pattern = r"\b[a-zA-Z_][a-zA-Z0-9_]*\b"
    return re.findall(pattern, f)


def is_valid_implication(
    assumptions: Sequence[Formula],
    conclusion: Formula,
    timeout: float | None = None,
) -> bool:
    """
    Check if the assumptions imply the conclusion.

    Only works for integer variables and returns `False` when in doubt.
    Works by verifying a WhyML program of this kind:

    ```mlw use int.Int

    let prog (x: int) (y: int) =
        assume {x >= 0}; assume {y >= 0}; assert {x + y >= 0}; ()
    ```
    """

    lines: list[str] = []
    indent: int = 0

    def add(line: str) -> None:
        nonlocal indent
        lines.append("    " * indent + line)

    all_fmls = [*assumptions, conclusion]
    vars = set([v for f in all_fmls for v in _get_variable_names(f)])
    args = " ".join(f"({v}: int)" for v in vars)

    add("use int.Int")
    add(f"let prog {args} = ")
    indent += 1
    for f in assumptions:
        add(f"assume {{{f}}};")
    add(f"assert {{{conclusion}}};")
    add("()")

    prog = "\n".join(lines)

    feedback = check(prog, prog, timeout=timeout)
    return feedback.success
