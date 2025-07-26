"""
Code2Inv Experiments
"""

import argparse
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
##### Experiments wih basic policy
#####


@dataclass
class Config:
    bench_name: str
    model_name: str
    temp: float
    num_concurrent: int
    max_requests: int
    seed: int


def experiment(config: Config):
    args: Any = {"prog": benchs[config.bench_name]}
    policy_args: Any = {
        "model_name": config.model_name,
        "temperature": config.temp,
        "num_concurrent": config.num_concurrent,
    }
    return cmd.RunStrategyArgs(  # type: ignore
        strategy="prove_program_by_recursive_abduction",
        args=args,
        policy="prove_program_by_saturation_basic_policy",
        policy_args=policy_args,
        num_generated=1,
        budget={dp.NUM_REQUESTS: config.max_requests},
    )


def make_code2inv_experiment(name: str, configs: Sequence[Config]):
    return exp.Experiment(
        name=name,
        dir=output_dir / name,
        context=context,
        experiment=experiment,
        config_type=Config,
        configs=configs,
        config_naming=lambda c, id: f"{c.bench_name}_{id}",
    )


#####
##### Experiments wih ensemble policy
#####


@dataclass
class EnsembleConfig:
    bench_name: str
    model_cycle: Sequence[tuple[str, int]]
    temperature: float
    num_concurrent: int
    max_requests_per_attempt: int
    max_dollar_budget: float
    seed: int


def ensemble_experiment(config: EnsembleConfig):
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
        policy="prove_program_by_saturation_ensemble_policy",
        policy_args=policy_args,
        num_generated=1,
        budget={dp.DOLLAR_PRICE: config.max_dollar_budget},
    )


def make_code2inv_ensemble_experiment(
    name: str, configs: Sequence[EnsembleConfig]
):
    return exp.Experiment(
        name=name,
        dir=output_dir / name,
        context=context,
        experiment=ensemble_experiment,
        config_type=EnsembleConfig,
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


#####
##### Creating CLIs
#####


def run_app(exp: exp.Experiment[Any]):
    parser = argparse.ArgumentParser(
        description="Run experiment with configurable max_workers"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=2,
        help="Maximum number of worker threads (default: 2)",
    )
    parser.add_argument(
        "--cache-requests",
        type=bool,
        default=True,
        help="Whether to cache requests (default: True)",
    )
    parser.add_argument(
        "--export-raw-trace",
        type=bool,
        default=False,
        help="Whether to export raw trace (default: False)",
    )
    parser.add_argument(
        "--export-log",
        type=bool,
        default=True,
        help="Whether to export log (default: True)",
    )
    parser.add_argument(
        "--export-browsable-trace",
        type=bool,
        default=False,
        help="Whether to export browsable trace (default: False)",
    )
    parser.add_argument(
        "--cache-only",
        action="store_true",
        help="Only export cache (sets cache_requests=True, all other exports=False)",
    )
    parser.add_argument(
        "--minimal-output",
        action="store_true",
        help="Minimal output mode (sets all export options to False)",
    )
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="Mark errors as todos to retry them",
    )
    args = parser.parse_args()

    # Handle special modes
    if args.cache_only:
        cache_requests = True
        export_raw_trace = False
        export_log = False
        export_browsable_trace = False
    elif args.minimal_output:
        cache_requests = False
        export_raw_trace = False
        export_log = False
        export_browsable_trace = False
    else:
        cache_requests = args.cache_requests
        export_raw_trace = args.export_raw_trace
        export_log = args.export_log
        export_browsable_trace = args.export_browsable_trace

    # Set experiment fields before loading
    exp.cache_requests = cache_requests
    exp.export_raw_trace = export_raw_trace
    exp.export_log = export_log
    exp.export_browsable_trace = export_browsable_trace

    exp.load()
    if args.retry_errors:
        exp.mark_errors_as_todos()
    exp.resume(max_workers=args.max_workers)
