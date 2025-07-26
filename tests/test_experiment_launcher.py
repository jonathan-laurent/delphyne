"""
Test the experiment launcher functionality.
"""

from dataclasses import dataclass
from pathlib import Path

import delphyne as dp
import delphyne.stdlib.commands as cmd
from delphyne.stdlib.experiments.experiment_launcher import quick_experiment


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


def test_experiment_launcher():
    configs = [
        MakeSumConfig(allowed=[1, 2, 3, 4, 5], goal=5),
        MakeSumConfig(allowed=[9, 6, 2], goal=11),
    ]
    experiment = quick_experiment(
        make_sum_experiment,
        configs,
        name="test_make_sum_experiment",
        workspace_root=Path(__file__).parent,
        modules=["example_strategies"],
        demo_files=[],
        output_dir=Path("output"),
    )
    experiment.load()
    experiment.get_status()
    # Verify the experiment was created
    assert experiment.dir.exists()
    assert experiment.dir.is_dir()
    # Check that configs were added
    state = experiment.load_state()
    assert state is not None
    assert len(state.configs) == 2
    # Run the experiment or replay if already done
    if experiment.is_done():
        print("Experiment already completed, replaying all configs...")
        experiment.replay_all_configs()
    else:
        print("Running experiment with 2 workers...")
        experiment.resume(max_workers=2, log_progress=False)
        print(f"Experiment completed successfully in {experiment.dir}")
    experiment.get_status()
    assert experiment.is_done()


if __name__ == "__main__":
    test_experiment_launcher()
