"""
Custom Delphyne Commands
"""

import traceback
from dataclasses import dataclass
from typing import Any

import delphyne.analysis as analysis
import delphyne.core as dp
import delphyne.stdlib.tasks as ta
import delphyne.utils.typing as ty
from delphyne.stdlib.commands import STD_COMMANDS


@dataclass
class CommandSpec:
    command: str
    args: dict[str, object]


async def execute_command(
    task: ta.TaskContext[ta.CommandResult[Any]],
    exe: ta.CommandExecutionContext,
    cmd: CommandSpec,
):
    try:
        loader = analysis.ObjectLoader(exe.base, extra_objects=STD_COMMANDS)
        command = loader.find_object(cmd.command)
        args_type = ta.command_args_type(command)
        args = ty.pydantic_load(args_type, cmd.args)
        await command(task, exe, args)
    except analysis.ObjectNotFound as e:
        error = ("error", f"Not found: {e}")
        await task.set_result(ta.CommandResult([error], None))
    except dp.InvalidDemoFile as e:
        error = ("error", f"Invalid demonstration file: {e.file}")
        await task.set_result(ta.CommandResult([error], None))
    except dp.TemplateError as e:
        error = ("error", f"Invalid prompt template `{e.name}`:\n{e.exn}")
        await task.set_result(ta.CommandResult([error], None))
    except dp.TemplateFileMissing as e:
        error = ("error", f"Prompt template file missing: {e.file}")
        await task.set_result(ta.CommandResult([error], None))
    except Exception as e:
        error = (
            "error",
            f"Internal error: {repr(e)}\n\n{traceback.format_exc()}",
        )
        await task.set_result(ta.CommandResult([error], None))
