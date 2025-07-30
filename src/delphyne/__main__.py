"""
Standard Command Line Tools for Delphyne
"""

import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import fire  # type: ignore
import yaml

import delphyne.stdlib as std
import delphyne.utils.typing as ty
from delphyne.scripts.command_utils import clear_command_file_outcome
from delphyne.scripts.demonstrations import check_demo_file
from delphyne.scripts.load_configs import load_config
from delphyne.server.execute_command import CommandSpec
from delphyne.utils.yaml import pretty_yaml


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

    def check(self, file: str):
        """
        Check a demo file.
        """
        file_path = Path(file)
        config = load_config(self.workspace_dir, local_config_from=file_path)
        feedback = check_demo_file(
            file_path, config.strategy_dirs, config.modules
        )
        self._process_diagnostics(feedback.warnings, feedback.errors)

    def run(
        self,
        file: str,
        no_output: bool = False,
        cache: bool = False,
        update: bool = False,
        header: bool = True,
    ):
        """
        Execute a command file.
        """
        file_path = Path(file)
        config = load_config(self.workspace_dir, local_config_from=file_path)
        if cache and not config.cache_root:
            config = replace(config, cache_root=file_path.parent / "cache")
        with open(file, "r") as f:
            spec = ty.pydantic_load(CommandSpec, yaml.safe_load(f))
        cmd, args = spec.load(config.base)
        if cache:
            assert hasattr(args, "cache_dir")
            assert hasattr(args, "cache_format")
            if not args.cache_dir:
                args.cache_dir = file_path.stem
            args.cache_format = "db"
        res = std.run_command(cmd, args, config)
        res_type = std.command_optional_result_wrapper_type(cmd)
        res_yaml = pretty_yaml(ty.pydantic_dump(res_type, res))
        if not no_output:
            print(res_yaml)
        if update:
            pass
        errors = [d[1] for d in res.diagnostics if d[0] == "error"]
        warnings = [d[1] for d in res.diagnostics if d[0] == "warning"]
        self._process_diagnostics(warnings, errors, use_stderr=not no_output)

    def clear(self, file: str):
        """
        Clear the outcome of a command file in place.
        """
        path = Path(file)
        with open(path, "r") as f:
            content = f.read()
        new_content = clear_command_file_outcome(content)
        with open(path, "w") as f:
            f.write(new_content)

    def serve(self):
        from delphyne.server.__main__ import main

        main()


def main():
    fire.Fire(DelphyneApp)  # type: ignore


if __name__ == "__main__":
    main()
