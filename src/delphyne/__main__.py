"""
Standard Command Line Tools for Delphyne
"""

import sys
from functools import partial
from pathlib import Path

import fire  # type: ignore
import yaml

import delphyne.stdlib as std
import delphyne.utils.typing as ty
from delphyne.scripts.demonstrations import check_demo_file
from delphyne.scripts.load_configs import load_config
from delphyne.server.execute_command import CommandSpec


class DelphyneApp:
    """
    Delphyne Command Line Application
    """

    def __init__(
        self,
        workspace_dir: Path | None = None,
        ensure_no_error: bool = False,
        ensure_no_warning: bool = False,
    ):
        self.workspace_dir = workspace_dir or Path.cwd()
        self.ensure_no_error = ensure_no_error
        self.ensure_no_warning = ensure_no_warning

    def _process_diagnostics(
        self, warnings: list[str], errors: list[str], use_stderr: bool = False
    ):
        show = partial(print, file=sys.stderr if use_stderr else sys.stdout)
        num_errors = len(errors)
        num_warnings = len(warnings)
        show(f"{num_errors} error(s), {num_warnings} warning(s)", end="\n\n")
        if errors or warnings:
            show("")
        for e in errors:
            show(f"Error: {e}")
        for w in warnings:
            show(f"Warning: {w}")
        if self.ensure_no_error and num_errors > 0:
            exit(1)
        if self.ensure_no_warning and num_warnings > 0:
            exit(1)

    def check_demo(self, file: Path):
        """
        Check a demo file.
        """
        config = load_config(self.workspace_dir, local_config_from=file)
        feedback = check_demo_file(file, config.strategy_dirs, config.modules)
        self._process_diagnostics(feedback.warnings, feedback.errors)

    def exec_command(
        self,
        file: Path,
        no_output: bool = False,
    ):
        """
        Execute a command file.
        """
        config = load_config(self.workspace_dir, local_config_from=file)
        with open(file, "r") as f:
            spec = ty.pydantic_load(CommandSpec, yaml.safe_load(f))
        cmd, args = spec.load(config.base)
        res = std.run_command(cmd, args, config)
        res_type = std.CommandResult[std.command_result_type(cmd) | None]
        res_yaml = yaml.dump(ty.pydantic_dump(res_type, res))
        if not no_output:
            print(res_yaml)
        errors = [d[1] for d in res.diagnostics if d[0] == "error"]
        warnings = [d[1] for d in res.diagnostics if d[0] == "warning"]
        self._process_diagnostics(warnings, errors, use_stderr=not no_output)


if __name__ == "__main__":
    fire.Fire(DelphyneApp)  # type: ignore
