"""
Standard Delphyne Commands
"""

# ruff: noqa: F401
# pyright: reportUnusedImport=false

from delphyne.stdlib.commands.answer_query import (
    AnswerQueryArgs,
    answer_query,
)
from delphyne.stdlib.commands.run_strategy import (
    RunStrategyArgs,
    run_strategy,
)
from delphyne.stdlib.tasks import (
    TestCommandArgs,
    test_command,
)

STD_COMMANDS: dict[str, object] = {
    "test_command": test_command,
    "run_strategy": run_strategy,
    "answer_query": answer_query,
}
