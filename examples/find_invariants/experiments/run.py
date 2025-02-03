"""
Run the experiments for the "Oracular Programming" paper.
"""

import pathlib

BENCHMARKS_FOLDER = pathlib.Path(__file__).parent.parent / "benchmarks" / "code2inv"
BENCHMARKS_MLW = BENCHMARKS_FOLDER / "mlw"

DEDUPLICATED = BENCHMARKS_FOLDER / "deduplicated.txt"
OUTPUT_FOLDER = pathlib.Path(__file__).parent / "out"


def load_benchmarks() -> list[tuple[str, str]]:
    with open(DEDUPLICATED, "r") as f:
        benchmarks = [line.strip() for line in f.readlines()]
    ret: list[tuple[str, str]] = []
    for name in benchmarks:
        with open(BENCHMARKS_MLW / (name + ".mlw"), "r") as f:
            content = f.read()
            ret.append((name, content))
    return ret


def evaluate_baseline(attempts: int = 32):
    pass


def generate_selected():
    benchs = load_benchmarks()
    selected = BENCHMARKS_FOLDER / "selected"
    selected.mkdir(exist_ok=True)
    for name, content in benchs:
        with open(selected / (name + ".mlw"), "w") as f:
            f.write(content)


if __name__ == "__main__":
    print(load_benchmarks())
    generate_selected()
    evaluate_baseline()
    pass
