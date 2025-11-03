from dataclasses import dataclass
from pathlib import Path
from typing import Any

import delphyne as dp
import yaml


def load_benchs() -> dict[str, Any]:
    tests_path = Path(__file__).parent / "tests.yaml"
    with open(tests_path, "r") as f:
        tests = yaml.safe_load(f)
    return {test["name_guess"]: test for test in tests}


BENCHS = load_benchs()


@dataclass
class FindTheoremConfig:
    problem: str
    model: str
    effort: dp.ReasoningEffort | str

    def instantiate(self, context: object):
        return dp.RunStrategyArgs(
            strategy="find_theorems",
            args={"request": BENCHS[self.problem]},
            policy="FindTheoremPolicy",
            policy_args={"model_name": self.model, "effort": self.effort},
            budget={},
        )


CONFIGS = [
    FindTheoremConfig(problem, model, effort)
    for problem in list(BENCHS.keys())[:1]
    for model, effort in [
        ("gpt-5-mini", "low"),
        # ("gpt-5", "low"),
    ]
]


if __name__ == "__main__":
    dp.Experiment(
        config_class=FindTheoremConfig,
        context=dp.workspace_execution_context(__file__),
        configs=CONFIGS,
        output_dir=Path(__file__).absolute().parent / "output",
    ).run_cli()
