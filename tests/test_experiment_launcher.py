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
from delphyne.stdlib.experiments.experiment_launcher import quick_experiment

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
    experiment = quick_experiment(
        exp_fun,
        configs,
        name=name,
        workspace_root=Path(__file__).parent,
        modules=["example_strategies"],
        demo_files=[],
        output_dir=Path("output"),
    )
    experiment.load()
    experiment.get_status()
    # Run the experiment or replay if already done
    if experiment.is_done():
        print("Experiment already completed, replaying all configs...")
        experiment.replay_all_configs()
    else:
        print("Running experiment...")
        experiment.resume(max_workers=2, log_progress=False)
        print(f"Experiment completed successfully in {experiment.dir}")
    experiment.get_status()
    assert experiment.is_done()
