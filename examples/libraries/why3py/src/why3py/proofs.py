import re
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass

from why3py import core

DEFAULT_MAX_STEPS = 5000
DEFAULT_MAX_TIME_IN_SECONDS = 5.0


def check_valid(
    src: str,
    *,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_time_in_seconds: float = DEFAULT_MAX_TIME_IN_SECONDS,
) -> str | None:
    """
    Take an annotated WhyML program and try to prove it automatically.
    Return `None` if and only if all obligations can be discharged and
    an error message otherwise.
    """
    match core.prove(src, max_steps, max_time_in_seconds):
        case ("Answer", (obligations,)):
            for obl in obligations:
                if not obl["proved"]:
                    return "Unproved obligation: " + obl["name"]
        case ("Error", (msg,)):
            return msg
    return None


def run_why3_ide(src: str) -> None:
    with tempfile.NamedTemporaryFile("w", suffix=".mlw") as f:
        f.write(src)
        f.flush()
        subprocess.run(["why3", "ide", f.name])


def clean_identifiers(s: str) -> str:
    """
    Why3 generates unique identifiers such as x5 or y34 by incrementing
    a counter. For redability, this function takes the text of an
    obligation and substitutes shorter identifiers, preserving their
    base names (x and y in the former example).
    """
    # We first find all identifiers for goals, constants or axioms
    # Identifiers can contain the "'" character
    identifiers = re.findall(r"(?:goal|constant|axiom) ((?:\w|')+) :", s)
    # We split out the numerical part of those identifiers
    bases: dict[str, list[str]] = defaultdict(list)
    for i in identifiers:
        m = re.match(r"([a-zA-Z_']+)(\d*)", i)
        assert m is not None
        base, num = m.groups()
        bases[base].append(num)
    # For each base name, we reassign numbers in increasing order
    conversions: dict[str, str] = {}
    for base, nums in bases.items():
        nums.sort(key=lambda n: int(n) if n else -1)
        if len(nums) == 1:
            conversions[base + nums[0]] = base
        else:
            for i, n in enumerate(nums):
                conversions[base + n] = f"{base}{i + 1}"
    # We apply our conversion table on each identifier
    return re.sub(
        r"(?:\w|')+", lambda m: conversions.get(m.group(), m.group()), s
    )


@dataclass
class Sequent:
    context: str
    goal: str


def split_sequent(sequent: str) -> Sequent:
    """
    Split the local context from the goal in a string representing a
    Why3 sequent.
    """
    match = re.search(
        r"-+ Local Context -+(.*?)-+ Goal -+(.*)",
        sequent,
        re.MULTILINE | re.DOTALL,
    )
    assert match is not None, "Could not split sequent"
    return Sequent(match.group(1).strip(), match.group(2).strip())
