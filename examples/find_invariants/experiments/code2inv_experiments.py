"""
Code2Inv Experiments
"""

from collections.abc import Sequence
from dataclasses import dataclass

import code2inv
import delphyne as dp
import delphyne.stdlib.commands as cmd

BENCHS = code2inv.load_all_benchmarks()
MODULES = ["abduct_and_saturate", "baseline"]
DEMO_FILES = MODULES


#####
##### Abduction Experiment
#####


@dataclass
class AbductionConfig:
    bench_name: str
    model_cycle: Sequence[tuple[str, int]]
    temperature: float
    num_concurrent: int
    max_requests_per_attempt: int
    max_dollar_budget: float
    seed: int


def abduction_experiment(config: AbductionConfig):
    return cmd.RunStrategyArgs(  # type: ignore
        strategy="prove_program_by_recursive_abduction",
        args={"prog": BENCHS[config.bench_name]},
        policy="prove_program_by_saturation",
        policy_args={
            "model_cycle": config.model_cycle,
            "temperature": config.temperature,
            "num_concurrent": config.num_concurrent,
            "max_requests_per_attempt": config.max_requests_per_attempt,
        },
        num_generated=1,
        budget={dp.DOLLAR_PRICE: config.max_dollar_budget},
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
    return cmd.RunStrategyArgs(  # type: ignore
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
