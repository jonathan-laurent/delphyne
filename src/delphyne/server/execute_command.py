"""
Custom Delphyne Commands
"""

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import delphyne.analysis as analysis
import delphyne.core as dp
import delphyne.stdlib.environments as en
import delphyne.stdlib.tasks as ta
import delphyne.utils.typing as ty
from delphyne.stdlib.commands import STD_COMMANDS


@dataclass
class CommandSpec:
    command: str
    args: dict[str, object]

    def load(
        self, ctx: analysis.DemoExecutionContext
    ) -> tuple[ta.Command[Any, Any], Any]:
        loader = analysis.ObjectLoader(ctx, extra_objects=STD_COMMANDS)
        command = loader.find_object(self.command)
        args_type = ta.command_args_type(command)
        args = ty.pydantic_load(args_type, self.args)
        return (command, args)


def execute_command(
    task: ta.TaskContext[ta.CommandResult[Any]],
    exe: ta.CommandExecutionContext,
    workspace_root: Path,
    cmd: CommandSpec,
):
    try:
        exe = exe.with_root(workspace_root)
        command, args = cmd.load(exe.base)
        command(task, exe, args)
    except analysis.ObjectNotFound as e:
        error = ("error", f"Not found: {e}")
        task.set_result(ta.CommandResult([error], None))
    except en.InvalidDemoFile as e:
        error = ("error", f"Invalid demonstration file: {e.file}")
        task.set_result(ta.CommandResult([error], None))
    except dp.TemplateError as e:
        error = ("error", f"Invalid prompt template `{e.name}`:\n{e.exn}")
        task.set_result(ta.CommandResult([error], None))
    except dp.TemplateFileMissing as e:
        error = ("error", f"Prompt template file missing: {e.file}")
        task.set_result(ta.CommandResult([error], None))
    except Exception as e:
        error = (
            "error",
            f"Internal error: {repr(e)}\n\n{traceback.format_exc()}",
        )
        task.set_result(ta.CommandResult([error], None))
