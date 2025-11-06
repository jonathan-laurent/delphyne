"""
A domain-specific language for writing declarative Lean proofs.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import assert_type

#####
##### Basic DSL Types
#####


type LeanTheorem = str
"""
A theorem statement with an empty proof, such as:

```
theorem my_theorem: 1 + 1 = 2 := by sorry
```
"""

type LeanExpr = str
"""
A Lean expression.
"""


type Identifier = str
"""
A string identifier for a Lean expression.
"""


type ProofSketchStep = Define | Prove | Suppose


type LeanProof = str


@dataclass
class ProofSketch:
    steps: Sequence[ProofSketchStep] = ()
    comment: str | None = None

    def num_holes(self) -> int:
        """
        Return the number of holes (i.e., unproven subgoals) in this
        proof sketch.
        """
        return _num_holes(self.steps)


@dataclass
class Define:
    define: tuple[Identifier, LeanExpr]
    comment: str | None = None


@dataclass
class Prove:
    prove: tuple[Identifier, LeanExpr]
    comment: str | None = None


@dataclass
class Suppose:
    suppose: tuple[Identifier, LeanExpr]
    do: Sequence[ProofSketchStep]
    conclude: tuple[Identifier, LeanExpr]
    comment: str | None = None


#####
##### Compilation
#####


def compile_sketch(
    theorem: LeanTheorem,
    sketch: ProofSketch,
    proofs: Sequence[LeanProof | None],
) -> str:
    """
    Compile a (partially filled) proof sketch into a Lean command.

    Attributes:
        theorem: The theorem statement. Everything after `by` is
            ignored.
        sketch: The proof sketch.
        proofs: A list of (optional) proofs for every hole in the
            sketch. Proofs are presented in the order they appear in the
            compiled sketch.
    """

    lines: list[str] = [*_extract_theorem_lines(theorem)]
    indent = 1
    cur_proofs = [*proofs]

    def write(line: str) -> None:
        # assert "\n" not in line
        lines.append("  " * indent + line)

    def write_proof(binding: tuple[str, str] | None):
        decl = f"have {binding[0]}: {binding[1]}" if binding else None
        nonlocal indent
        assert cur_proofs, "Not enough proofs provided."
        proof = cur_proofs.pop(0)
        if decl is None:
            if proof is None:
                write("sorry")
            else:
                for proof_line in proof.splitlines():
                    write(proof_line)
        else:
            if proof is None:
                write(f"{decl} := by sorry")
            else:
                write(f"{decl} := by")
                indent += 1
                for proof_line in proof.splitlines():
                    write(proof_line)
                indent -= 1

    def compile_step(step: ProofSketchStep) -> None:
        nonlocal indent
        if isinstance(step, Define):
            rhs = _one_liner(step.define[1])
            write(f"let {step.define[0]} := {rhs}")
        elif isinstance(step, Prove):
            write_proof(step.prove)
        else:
            assert_type(step, Suppose)
            assum_name = step.suppose[0]
            assum = _one_liner(step.suppose[1])
            concl_name = step.conclude[0]
            concl = _one_liner(step.conclude[1])
            proved = f"({assum}) → {concl}"
            write(f"have {concl_name}: {proved} := by")
            indent += 1
            write(f"intro {assum_name}")
            for substep in step.do:
                compile_step(substep)
            write_proof(None)
            indent -= 1

    for step in sketch.steps:
        compile_step(step)
    write_proof(None)
    assert not cur_proofs, "Too many proofs provided."

    return "\n".join(lines)


def _num_holes(sketch: Sequence[ProofSketchStep]) -> int:
    count = 0
    for step in sketch:
        match step:
            case Prove():
                count += 1
            case Suppose(do=do_steps):
                count += _num_holes(do_steps)
            case Define():
                pass
    # +1 for the final hole that establishes the conclusion after
    # assuming all the steps.
    return count + 1


#####
##### Utils
#####


def _one_liner(s: str) -> str:
    """
    Replace all consecutive spaces by single-spaces, and trim leading
    and trailing spaces.
    """
    import re

    # Replace consecutive whitespace with single spaces and trim
    return re.sub(r"\s+", " ", s).strip()


def _extract_theorem_lines(theorem: LeanTheorem) -> list[str]:
    """
    Take a Lean theorem and return the theorem statement as a list of
    lines. The last line is not empty, and the proof that follows the
    first occurence of the `by` keyword is dicarded.
    """
    lines = theorem.split("\n")
    result_lines: list[str] = []

    for line in lines:
        # Check if this line contains 'by'
        if " by" in line:
            # Find the position of 'by' and truncate the line there
            by_index = line.find(" by")
            truncated_line = line[
                : by_index + 3
            ].rstrip()  # +3 to include ' by'
            if truncated_line.strip():  # Only add non-empty lines
                result_lines.append(truncated_line)
            break
        else:
            # Add the line if it's not empty when stripped
            if line.strip():
                result_lines.append(line)

    assert result_lines[-1].strip()
    return result_lines
