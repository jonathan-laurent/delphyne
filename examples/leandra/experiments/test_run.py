from dataclasses import dataclass, replace
from pathlib import Path

import minif2f
from learning_experiments import workers_setup

import delphyne as dp
from delphyne.stdlib.experiments.experiment_launcher import ExperimentCLI

TESTING_THEOREMS = minif2f.load_minif2f("test")
INIT_COMMANDS = ["import MiniF2F.Minif2fImport"]


@dataclass
class Config:
    bench_name: str
    budget: float

    def instantiate(self, context: object):
        return dp.RunStrategyArgs(
            strategy="prove_theorem",
            args={"theorem": TESTING_THEOREMS[self.bench_name]},
            policy="ProveTheoremPolicy",
            policy_args={"only_permanent_examples": True},
            budget={
                dp.DOLLAR_PRICE: self.budget,
            },
        )


class Experiment(ExperimentCLI):
    def __init__(
        self,
        name: str = "default",
        n: int | None = None,
        budget: float = 1.0,
    ) -> None:
        """
        Arguments:
            name: Name of the experiment.
            n: Number of problems to include in the experiment.
            budget: Budget (in dollars) allocated to each problem.
        """
        theorems = minif2f.load_minif2f("test")
        if n is None:
            problems = list(theorems.keys())
        else:
            problems = [name for name in theorems.keys()][:n]
        all_outputs = Path(__file__).absolute().parent / "output"
        output_direcory =  all_outputs / "test_run" / name
        context = dp.workspace_execution_context(__file__)
        context = replace(context, init=())
        experiment = dp.Experiment(
            config_class=Config,
            context=context,
            configs=[Config(bench, budget) for bench in problems],
            output_dir=output_direcory,
            workers_setup=workers_setup,
        )
        super().__init__(experiment)


if __name__ == "__main__":
    import fire  # type: ignore

    fire.Fire(Experiment)   # type: ignore