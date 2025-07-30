from pathlib import Path
from shutil import rmtree

import delphyne as dp
import delphyne.stdlib.commands as cmd
import delphyne.stdlib.tasks as ta

TEMP_OUT_DIR = Path(__file__).parent / "cmd_out"
STRATEGY_FILE = "example_strategies"
TESTS_FOLDER = Path(__file__).parent


def test_counting_command(delay: float = 1e-4):
    rmtree(TEMP_OUT_DIR, ignore_errors=True)
    dp.run_command(
        ta.test_command,
        ta.TestCommandArgs(10, delay=delay),
        dp.CommandExecutionContext(),
        dump_statuses=TEMP_OUT_DIR / "counting_statuses.txt",
        dump_result=TEMP_OUT_DIR / "counting_result.yaml",
        dump_log=TEMP_OUT_DIR / "counting_log.txt",
        on_status=lambda s: print(s, end="\r"),
    )
    rmtree(TEMP_OUT_DIR, ignore_errors=True)


def test_run_strategy():
    rmtree(TEMP_OUT_DIR, ignore_errors=True)
    dp.run_command(
        cmd.run_strategy,
        cmd.RunStrategyArgs(
            strategy="test_cached_computations",
            args={"n": 1},
            num_generated=1,
            policy="test_cached_computations_policy",
            policy_args={},
            budget={},
        ),
        dp.CommandExecutionContext(
            modules=[STRATEGY_FILE],
        ).with_root(TESTS_FOLDER),
        dump_statuses=TEMP_OUT_DIR / "strategy_statuses.txt",
        dump_result=TEMP_OUT_DIR / "strategy_result.yaml",
        dump_log=TEMP_OUT_DIR / "strategy_log.txt",
    )
    rmtree(TEMP_OUT_DIR, ignore_errors=True)


if __name__ == "__main__":
    test_counting_command(delay=1)
