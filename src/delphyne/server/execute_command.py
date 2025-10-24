"""
Custom Delphyne Commands
"""

import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import delphyne.analysis as analysis
import delphyne.analysis.feedback as fb
import delphyne.core_and_base as dp
import delphyne.stdlib.tasks as ta
import delphyne.utils.typing as ty
from delphyne.stdlib.commands import STD_COMMANDS


@dataclass
class CommandSpec:
    command: str
    args: dict[str, object]

    def load(
        self, object_loader: dp.ObjectLoader
    ) -> tuple[ta.Command[Any, Any], Any]:
        command = object_loader.find_object(self.command)
        args_type = ta.command_args_type(command)
        args = ty.pydantic_load(args_type, self.args)
        return (command, args)


def execute_command(
    task: ta.TaskContext[ta.CommandResult[Any]],
    exe: ta.ExecutionContext,
    workspace_root: Path,
    cmd: CommandSpec,
):
    try:
        exe = exe.with_root(workspace_root)
        loader = exe.object_loader(extra_objects=STD_COMMANDS)
        command, args = cmd.load(loader)
        command(task, exe, args)
    except analysis.ObjectNotFound as e:
        error = fb.Diagnostic("error", f"Not found: {e}")
        task.set_result(ta.CommandResult([error], None))
    except dp.InvalidDemoFile as e:
        error = fb.Diagnostic("error", f"Invalid demonstration file: {e.file}")
        task.set_result(ta.CommandResult([error], None))
    except dp.TemplateError as e:
        msg = f"Invalid prompt template `{e.name}`:\n{e.exn}"
        error = fb.Diagnostic("error", msg)
        task.set_result(ta.CommandResult([error], None))
    except dp.TemplateFileMissing as e:
        msg = f"Prompt template file missing: {e.file}"
        error = dp.Diagnostic("error", msg)
        task.set_result(ta.CommandResult([error], None))
    except Exception as e:
        error = fb.Diagnostic(
            "error",
            f"Internal error: {repr(e)}\n\n{traceback.format_exc()}",
        )
        task.set_result(ta.CommandResult([error], None))
