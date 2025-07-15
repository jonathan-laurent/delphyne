"""
Utilities for loading the Code2Inv benchmark problems.
"""

import pathlib

BENCHMARKS_FOLDER = (
    pathlib.Path(__file__).parent.parent / "benchmarks" / "code2inv"
)
BENCHMARKS_MLW = BENCHMARKS_FOLDER / "mlw"
WRONG = BENCHMARKS_FOLDER / "wrong.txt"
OUTPUT_FOLDER = pathlib.Path(__file__).parent / "out"


def load_blacklist() -> set[str]:
    with open(WRONG, "r") as f:
        lines = f.readlines()
    return {line.strip() for line in lines}


def load_all_benchmarks() -> dict[str, str]:
    """
    Load all benchmarks from the code2inv folder.
    """
    blacklist = load_blacklist()
    ret: dict[str, str] = {}
    for path in BENCHMARKS_MLW.glob("*.mlw"):
        name = path.stem
        if name in blacklist:
            continue
        with open(path, "r") as f:
            content = f.read()
            ret[name] = content
    return ret


if __name__ == "__main__":
    print(len(load_all_benchmarks()))
