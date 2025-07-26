"""
Code2Inv Experiments
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import code2inv
import delphyne as dp
import delphyne.stdlib.commands as cmd
from delphyne.stdlib.experiments import experiment_launcher as exp

output_dir = Path(__file__).parent / "output"
root = Path(__file__).parent.parent
strategy_dirs = [root]
modules = ["why3_utils", "baseline", "abduct_and_saturate"]
demo_files = [
    root / "baseline.demo.yaml",
    root / "abduct_and_saturate.demo.yaml",
]
prompt_dirs = [root / "prompts"]
demo_context = dp.DemoExecutionContext(strategy_dirs, modules)
context = dp.CommandExecutionContext(
    demo_context,
    demo_files,
    prompt_dirs,
    data_dirs=[],
    result_refresh_period=None,
    status_refresh_period=None,
)
benchs = code2inv.load_all_benchmarks()


#####
##### Experiments with the Abduction-Based Agent
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
    args: Any = {"prog": benchs[config.bench_name]}
    policy_args: Any = {
        "model_cycle": config.model_cycle,
        "temperature": config.temperature,
        "num_concurrent": config.num_concurrent,
        "max_requests_per_attempt": config.max_requests_per_attempt,
    }
    return cmd.RunStrategyArgs(  # type: ignore
        strategy="prove_program_by_recursive_abduction",
        args=args,
        policy="prove_program_by_saturation",
        policy_args=policy_args,
        num_generated=1,
        budget={dp.DOLLAR_PRICE: config.max_dollar_budget},
    )


def make_code2inv_abduction_experiment(
    name: str, configs: Sequence[AbductionConfig]
):
    return exp.Experiment(
        name=name,
        dir=output_dir / name,
        context=context,
        experiment=abduction_experiment,
        config_type=AbductionConfig,
        configs=configs,
        config_naming=lambda c, id: f"{c.bench_name}_{id}",
    )


#####
##### Baseline Experiments
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
    args: Any = {"prog": benchs[config.bench_name]}
    policy_args: Any = {
        "model_name": config.model_name,
        "temperature": config.temperature,
        "max_feedback_cycles": config.max_feedback_cycles,
        "loop": config.loop,
    }
    budget: dict[str, float] = {}
    if config.max_dollar_budget is not None:
        budget[dp.DOLLAR_PRICE] = config.max_dollar_budget
    return cmd.RunStrategyArgs(  # type: ignore
        strategy="prove_program_interactive",
        args=args,
        policy="prove_program_interactive_policy",
        policy_args=policy_args,
        num_generated=1,
        budget=budget,
    )


def make_code2inv_baseline_experiment(
    name: str, configs: Sequence[BaselineConfig]
):
    return exp.Experiment(
        name=name,
        dir=output_dir / name,
        context=context,
        experiment=baseline_experiment,
        config_type=BaselineConfig,
        configs=configs,
        config_naming=lambda c, id: f"{c.bench_name}_{c.model_name}_{id}",
    )
