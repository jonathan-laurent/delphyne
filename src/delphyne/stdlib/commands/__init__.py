"""
Standard Delphyne Commands
"""

from delphyne.stdlib.commands.answer_query import answer_query
from delphyne.stdlib.commands.run_strategy import run_strategy
from delphyne.stdlib.tasks import test_command

STD_COMMANDS: dict[str, object] = {
    "test_command": test_command,
    "run_strategy": run_strategy,
    "answer_query": answer_query,
}
