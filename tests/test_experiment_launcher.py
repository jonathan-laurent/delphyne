"""
Test the experiment launcher functionality.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

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

    def instantiate(self, context: object):
        return cmd.RunStrategyArgs(
            strategy="make_sum",
            args={"allowed": self.allowed, "goal": self.goal},
            policy="make_sum_policy",
            policy_args={},
            num_generated=1,
            budget={dp.DOLLAR_PRICE: 0.05},  # Small budget for testing
        )


@dataclass
class CachedComputationsConfig:
    def instantiate(self, context: object):
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
    "name,config_class,configs",
    [
        (
            "test_make_sum_experiment",
            MakeSumConfig,
            [
                MakeSumConfig(allowed=[1, 2, 3, 4, 5], goal=5),
                MakeSumConfig(allowed=[9, 6, 2], goal=11),
            ],
        ),
        (
            "test_cached_computation_experiment",
            CachedComputationsConfig,
            [CachedComputationsConfig()],
        ),
    ],
)
def test_experiment_launcher[C: dp.ExperimentConfig](
    name: str,
    config_class: type[C],
    configs: Sequence[C],
):
    root = Path(__file__).parent
    context = dp.ExecutionContext(
        modules=["example_strategies"],
    ).with_root(root)
    experiment = dp.Experiment[C](
        config_class=config_class,
        context=context,
        configs=configs,
        name=name,
        output_dir=f"output/{name}",
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
