"""
Standard Command Line Tools for Delphyne
"""

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

    def __init__(self, workspace_dir: Path | None = None):
        self.workspace_dir = workspace_dir or Path.cwd()

    def check_demo(
        self,
        file: Path,
        ensure_no_error: bool = False,
        ensure_no_warning: bool = False,
    ):
        """
        Check a demo file.
        """
        config = load_config(self.workspace_dir)
        feedback = check_demo_file(file, config.strategy_dirs, config.modules)
        num_errors = len(feedback.errors)
        num_warnings = len(feedback.warnings)
        print(f"{num_errors} error(s), {num_warnings} warning(s)", end="\n\n")
        if feedback.errors or feedback.warnings:
            print("")
        for e in feedback.errors:
            print(f"Error: {e}")
        for w in feedback.warnings:
            print(f"Warning: {w}")
        if ensure_no_error and num_errors > 0:
            exit(1)
        if ensure_no_warning and num_warnings > 0:
            exit(1)

    def exec_command(self, file: Path, no_output: bool = False):
        """
        Execute a command file.
        """
        config = load_config(self.workspace_dir)
        with open(file, "r") as f:
            spec = ty.pydantic_load(CommandSpec, yaml.safe_load(f))
        cmd, args = spec.load(config.base)
        res = std.run_command(cmd, args, config)
        res_type = std.CommandResult[std.command_result_type(cmd) | None]
        res_yaml = yaml.dump(ty.pydantic_dump(res_type, res))
        if not no_output:
            print(res_yaml)


if __name__ == "__main__":
    fire.Fire(DelphyneApp)  # type: ignore
