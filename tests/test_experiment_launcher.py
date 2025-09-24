"""
Test the experiment launcher functionality.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import delphyne as dp
import delphyne.stdlib.commands as cmd

#####
##### Experiment Examples
#####


@dataclass
class MakeSumConfig:
    allowed: list[int]
    goal: int


def make_sum_experiment(config: MakeSumConfig) -> cmd.RunStrategyArgs:
    return cmd.RunStrategyArgs(
        strategy="make_sum",
        args={"allowed": config.allowed, "goal": config.goal},
        policy="make_sum_policy",
        policy_args={},
        num_generated=1,
        budget={dp.DOLLAR_PRICE: 0.05},  # Small budget for testing
    )


@dataclass
class CachedComputationsConfig:
    pass


def cached_computations_experiment(config: object):
    return cmd.RunStrategyArgs(
        strategy="test_cached_computations",
        args={"n": 3},
        policy="test_cached_computations_policy",
        policy_args={},
        num_generated=1,
        budget={dp.DOLLAR_PRICE: 0},
    )


#####
##### Harness
#####


def _init_worker(msg: str):
    import os

    print(f"Initialized worker ({os.getpid()}): {msg}")


@pytest.mark.parametrize(
    "name,exp_fun,configs",
    [
        (
            "test_make_sum_experiment",
            make_sum_experiment,
            [
                MakeSumConfig(allowed=[1, 2, 3, 4, 5], goal=5),
                MakeSumConfig(allowed=[9, 6, 2], goal=11),
            ],
        ),
        (
            "test_cached_computation_experiment",
            cached_computations_experiment,
            [CachedComputationsConfig()],
        ),
    ],
)
def test_experiment_launcher(
    name: str,
    exp_fun: Callable[[Any], cmd.RunStrategyArgs],
    configs: Sequence[Any],
):
    root = Path(__file__).parent
    context = dp.CommandExecutionContext(
        modules=["example_strategies"],
    ).with_root(root)
    experiment = dp.Experiment(
        experiment=exp_fun,
        context=context,
        configs=configs,
        name=name,
        output_dir=root / "output" / name,
        workers_setup=dp.WorkersSetup(
            common=lambda: "Hello World", per_worker=_init_worker
        ),
    )
    experiment.load()
    experiment.get_status()
    if experiment.is_done():
        print("Experiment already completed, replaying all configs...")
        experiment.replay_all_configs()
    else:
        print("Running experiment...")
        experiment.resume(max_workers=2, log_progress=False)
        print("Experiment completed successfully.")
    experiment.get_status()
    assert experiment.is_done()
    experiment.replay_all_configs()
