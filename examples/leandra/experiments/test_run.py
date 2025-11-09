from dataclasses import dataclass
from pathlib import Path

import delphyne as dp
import minif2f

TESTING_THEOREMS = minif2f.load_minif2f("test")
INIT_COMMANDS = ["import MiniF2F.Minif2fImport"]
NUM_PROBLEMS: int | None = None
DOLLAR_BUDGET = 1


@dataclass
class Config:
    bench_name: str

    def instantiate(self, context: object):
        return dp.RunStrategyArgs(
            strategy="prove_theorem",
            args={"theorem": TESTING_THEOREMS[self.bench_name]},
            policy="ProveTheoremPolicy",
            policy_args={"only_permanent_examples": True},
            budget={
                dp.DOLLAR_PRICE: DOLLAR_BUDGET,
            },
        )


if __name__ == "__main__":
    if NUM_PROBLEMS is None:
        problems = list(TESTING_THEOREMS.keys())
    else:
        problems = [name for name in TESTING_THEOREMS.keys()][:NUM_PROBLEMS]
    output_directory = Path(__file__).absolute().parent / "output" / "test_run"
    dp.Experiment(
        config_class=Config,
        context=dp.workspace_execution_context(__file__),
        configs=[Config(problems) for problems in problems],
        output_dir=f"experiments/output/{dp.path_stem(__file__)}",
    ).run_cli()
