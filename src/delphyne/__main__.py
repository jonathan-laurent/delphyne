"""
Standard Command Line Tools for Delphyne
"""

import sys
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Any

import fire  # type: ignore
import yaml

import delphyne.core as dp
import delphyne.stdlib as std
import delphyne.utils.typing as ty
from delphyne.scripts.command_utils import command_file_header
from delphyne.scripts.demonstrations import check_demo_file
from delphyne.scripts.load_configs import find_workspace_dir, load_config
from delphyne.server.execute_command import CommandSpec
from delphyne.utils.misc import StatusIndicator
from delphyne.utils.yaml import pretty_yaml

STATUS_REFRESH_PERIOD_IN_SECONDS = 1.0


class DelphyneCLI:
    """
    The Delphyne Command Line Interface.

    The `delphyne` package features a `delphyne` command line
    application that is automatically generated from the `DelphyneCLI`
    class using [fire](https://github.com/google/python-fire). In
    particular, this application can be used to check demonstration
    files, execute command files, and launch the Delphyne language
    server.
    """

    def __init__(
        self,
        *,
        workspace_dir: Path | None = None,
        ensure_no_error: bool = False,
        ensure_no_warning: bool = False,
    ):
        """
        Arguments:
            workspace_dir: The workspace directory. If not provided, it
                is deduced for each demonstration or command file by
                considering the closest transitive parent directory that
                contains a `delphyne.yaml` file. If no such directory
                exists, the current working directory is used.
            ensure_no_error: Exit with a nonzero code if an error is
                produced.
            ensure_no_warning: Exit with a nonzero code if a warning is
                produced.
        """
        self.workspace_dir = workspace_dir
        self.ensure_no_error = ensure_no_error
        self.ensure_no_warning = ensure_no_warning

    def _process_diagnostics(
        self,
        warnings: list[str],
        errors: list[str],
        use_stderr: bool = False,
        show_summary: bool = True,
    ):
        show = partial(print, file=sys.stderr if use_stderr else sys.stdout)
        num_errors = len(errors)
        num_warnings = len(warnings)
        if show_summary:
            show(f"{num_errors} error(s), {num_warnings} warning(s)")
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

    def _workspace_dir_for(self, file: Path) -> Path:
        """
        Get the workspace directory for a given file.
        """
        workspace_dir = self.workspace_dir
        if workspace_dir is None:
            workspace_dir = find_workspace_dir(file)
        if workspace_dir is None:
            workspace_dir = Path.cwd()
        return workspace_dir

    def check(self, file: str):
        """
        Check a demonstration file.
        """
        file_path = Path(file)
        workspace_dir = self._workspace_dir_for(file_path)
        config = load_config(workspace_dir, local_config_from=file_path)
        feedback = check_demo_file(
            file_path, config.strategy_dirs, config.modules
        )
        self._process_diagnostics(feedback.warnings, feedback.errors)

    def run(
        self,
        file: str,
        *,
        cache: bool = False,
        update: bool = False,
        no_output: bool = False,
        no_header: bool = False,
        no_status: bool = False,
        filter: list[str] | None = None,
        log_level: dp.LogLevel | None = None,
        clear: bool = False,
    ):
        """
        Execute a command file.

        Print an updated command file with an `outcome` section added on
        stdout. Print other information on stderr.

        Arguments:
            file: Path to the command file to execute.
            cache: Enable caching (assuming the command supports it).
            update: Update the command file in place with the outcome.
            no_output: Do not print on stdout.
            no_header: Only print the `outcome` section on stdout.
            no_status: Do not show the progress bar.
            filter: Only show the provided subset of fields for the
                `outcome.result` section.
            log_level: If provided and if the command supports it,
                overrides the command's `log_level` argument.
            clear: When this option is passed, all other options are
                ignored and the `clear` method is called.
        """
        file_path = Path(file)
        workspace_dir = self._workspace_dir_for(file_path)
        config = load_config(workspace_dir, local_config_from=file_path)
        config = replace(
            config,
            status_refresh_period=STATUS_REFRESH_PERIOD_IN_SECONDS,
            result_refresh_period=None,
        )
        if clear:
            self.clear(file)
            return
        if update:
            no_output = True
            no_header = False
        if cache and not config.cache_root:
            config = replace(config, cache_root=file_path.parent / "cache")
        with open(file, "r") as f:
            spec = ty.pydantic_load(CommandSpec, yaml.safe_load(f))
        cmd, args = spec.load(config.base)
        if cache:
            assert hasattr(args, "cache_file"), (
                "Command does not have a `cache_file` argument."
            )
            if not args.cache_file:
                args.cache_file = file_path.stem + ".yaml"
        if log_level:
            if not hasattr(args, "log_level"):
                raise ValueError(
                    "Command does not have a `log_level` argument."
                )
            assert dp.valid_log_level(log_level), (
                f"Invalid log level: {log_level}"
            )
            args.log_level = log_level
        progress = StatusIndicator(sys.stderr, show=not no_status)
        res = std.run_command(cmd, args, config, on_status=progress.on_status)
        progress.done()
        res_type = std.command_optional_result_wrapper_type(cmd)
        res_python: Any = ty.pydantic_dump(res_type, res)
        if filter and res_python["result"] is not None:
            res_python["result"] = {
                k: v for k, v in res_python["result"].items() if k in filter
            }
        if no_header:
            output = pretty_yaml(res_python)
        else:
            with open(file_path, "r") as f:
                header = command_file_header(f.read())
            output = header.rstrip() + "\n"
            output += pretty_yaml({"outcome": res_python})
        if not no_output:
            print(output)
        if update:
            with open(file_path, "w") as f:
                f.write(output)
        errors = [d[1] for d in res.diagnostics if d[0] == "error"]
        warnings = [d[1] for d in res.diagnostics if d[0] == "warning"]
        self._process_diagnostics(
            warnings,
            errors,
            use_stderr=True,
            show_summary=self.ensure_no_error or self.ensure_no_warning,
        )

    def clear(self, file: str):
        """
        Clear the outcome of a command file by updating it in place.
        """
        path = Path(file)
        with open(path, "r") as f:
            content = f.read()
        new_content = command_file_header(content)
        with open(path, "w") as f:
            f.write(new_content)

    def serve(self, *, port: int = 3008):
        """
        Launch an instance of the Delphyne language server.
        """
        from delphyne.server.__main__ import main

        main(port=port)


def main():
    fire.Fire(DelphyneCLI)  # type: ignore


if __name__ == "__main__":
    main()
