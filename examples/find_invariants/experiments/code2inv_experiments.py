"""
Code2Inv Experiments
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import code2inv
import delphyne as dp
import delphyne.stdlib.commands as cmd
from delphyne.stdlib.experiments.experiment_launcher import (
    Experiment,
    ExperimentFun,
)

BENCHS = code2inv.load_all_benchmarks()
MODULES = ["abduct_and_saturate", "baseline"]
DEMO_FILES = [Path(m) for m in MODULES]


def make_experiment[C](
    experiment: ExperimentFun[C],
    configs: Sequence[C],
    output_dir: str,
    exp_file: str,
) -> Experiment[C]:
    workspace_root = Path(exp_file).parent.parent
    exp_name = Path(exp_file).stem
    context = dp.CommandExecutionContext(
        modules=MODULES, demo_files=DEMO_FILES
    ).with_root(workspace_root)
    return Experiment(
        experiment=experiment,
        context=context,
        configs=configs,
        name=exp_name,
        output_dir=workspace_root / "experiments" / output_dir / exp_name,
    )


#####
##### Abduction Experiment
#####


@dataclass
class AbductionConfig:
    bench_name: str
    model_cycle: Sequence[tuple[str, int]]
    temperature: float
    num_completions: int
    max_requests_per_attempt: int
    max_dollar_budget: float
    max_retries_per_step: int
    max_propagation_steps: int
    seed: int


def abduction_experiment(config: AbductionConfig):
    return cmd.RunStrategyArgs(
        strategy="prove_program_by_recursive_abduction",
        args={"prog": BENCHS[config.bench_name]},
        policy="prove_program_by_saturation",
        policy_args={
            "model_cycle": config.model_cycle,
            "temperature": config.temperature,
            "num_completions": config.num_completions,
            "max_requests_per_attempt": config.max_requests_per_attempt,
        },
        num_generated=1,
        budget={dp.DOLLAR_PRICE: config.max_dollar_budget},
        log_long_computations=("info", 1.0),
    )


#####
##### Baseline Experiment
#####


@dataclass
class BaselineConfig:
    bench_name: str
    model_name: str
    temperature: float
    max_feedback_cycles: int
    seed: int
    loop: bool = False
    max_dollar_budget: float | None = 0.2


def baseline_experiment(config: BaselineConfig):
    budget: dict[str, float] = {}
    if config.max_dollar_budget is not None:
        budget[dp.DOLLAR_PRICE] = config.max_dollar_budget
    return cmd.RunStrategyArgs(
        strategy="prove_program_interactive",
        args={"prog": BENCHS[config.bench_name]},
        policy="prove_program_interactive_policy",
        policy_args={
            "model_name": config.model_name,
            "temperature": config.temperature,
            "max_feedback_cycles": config.max_feedback_cycles,
            "loop": config.loop,
        },
        num_generated=1,
        budget=budget,
    )
