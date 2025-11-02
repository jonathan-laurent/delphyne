"""
Utilities to work with the MiniF2F benchmark suite.
"""

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

type Split = Literal["test", "valid"]
"""
MiniF2F dataset splits.
"""


type Benchmark = dict[str, str]
"""
Mapping problem names to theorem commands with a `sorry` proof.
"""


def load_minif2f_file(split: Split):
    delphyne_repo = Path(__file__).absolute().parent.parent
    repo = delphyne_repo / "benchmarks" / "minif2f"
    file = "Test.lean" if split == "test" else "Valid.lean"
    path = repo / "MiniF2F" / file
    with open(path, "r") as f:
        return f.read()


def load_theorems(file_content: str) -> Sequence[str]:
    """
    Parse theorem blocks from a Lean file and return them as a list of
    strings.
    """
    # Split on 'theorem' and filter out empty parts
    parts = file_content.split("theorem ")
    theorems: list[str] = []

    for part in parts[1:]:  # Skip first part (before first theorem)
        theorem = "theorem " + part
        # Find where this theorem ends (next theorem or end of file)
        next_theorem_pos = theorem.find(
            "\ntheorem ", 8
        )  # Start after 'theorem '
        if next_theorem_pos != -1:
            theorem = theorem[:next_theorem_pos]
        theorems.append(theorem.strip())

    return theorems


def remove_proof(theorem: str) -> str:
    """
    Removesthe proof content from a theorem statement and replace it
    with ':= by sorry'.
    """
    by_pos = theorem.find(":= by")
    if by_pos == -1:
        return theorem
    theorem_signature = theorem[:by_pos].strip()
    return theorem_signature + " := by sorry"


def extract_theorem_name(theorem: str) -> str:
    """
    Extract the theorem name from a theorem statement.
    """
    parts = theorem.split("theorem ", 1)
    if len(parts) < 2:
        return ""
    return parts[1].split()[0]


def load_minif2f(split: Split) -> Benchmark:
    """
    Load the MiniF2F benchmark for the given split.

    Returns a dictionary mapping theorem names to theorem statements
    with proofs replaced by 'sorry'.
    """
    file_content = load_minif2f_file(split)
    theorems = load_theorems(file_content)

    benchmark: Benchmark = {}
    for theorem in theorems:
        theorem_without_proof = remove_proof(theorem)
        theorem_name = extract_theorem_name(theorem)
        assert theorem_name
        benchmark[theorem_name] = theorem_without_proof

    return benchmark


if __name__ == "__main__":
    print(load_minif2f("test"))
    print(load_minif2f("valid"))
