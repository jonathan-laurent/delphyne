from pathlib import Path
from shutil import rmtree

import delphyne as dp
import delphyne.stdlib.tasks as ta

TEMP_OUT_DIR = Path(__file__).parent / "cmd_out"


def test_counting_command():
    rmtree(TEMP_OUT_DIR, ignore_errors=True)
    dp.run_command(
        ta.test_command,
        ta.TestCommandArgs(10, delay=1e-4),
        dp.CommandExecutionContext(dp.DemoExecutionContext([], []), []),
        dump_statuses=TEMP_OUT_DIR / "statuses.txt",
        dump_result=TEMP_OUT_DIR / "result.yaml",
        dump_log=TEMP_OUT_DIR / "log.txt",
    )
    rmtree(TEMP_OUT_DIR, ignore_errors=True)
