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

import delphyne.analysis.feedback as fb
import delphyne.core as dp
import delphyne.stdlib as std
import delphyne.utils.typing as ty
from delphyne.scripts.command_utils import command_file_header
from delphyne.scripts.demonstrations import check_demo_file
from delphyne.server.execute_command import CommandSpec
from delphyne.stdlib.commands import STD_COMMANDS
from delphyne.stdlib.execution_contexts import (
    load_execution_context,
    surrounding_workspace_dir,
)
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
            workspace_dir = surrounding_workspace_dir(file)
        if workspace_dir is None:
            workspace_dir = Path.cwd()
        return workspace_dir

    def check(self, file: str):
        """
        Check a demonstration file.
        """
        file_path = Path(file)
        workspace_dir = self._workspace_dir_for(file_path)
        config = load_execution_context(workspace_dir, local=file_path)
        feedback = check_demo_file(file_path, config, workspace_dir)
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
        config = load_execution_context(workspace_dir, local=file_path)
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
        loader = config.object_loader(extra_objects=STD_COMMANDS)
        cmd, args = spec.load(loader)
        if cache:
            assert hasattr(args, "cache_file"), (
                "Command does not have a `cache_file` argument."
            )
            if not args.cache_file:
                args.cache_file = file_path.stem + ".yaml"
            assert hasattr(args, "embeddings_cache_file"), (
                "Command does not have an `embeddings_cache_file` argument."
            )
            if not args.embeddings_cache_file:
                args.embeddings_cache_file = file_path.stem + ".embeddings.h5"
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
        errors = [d.message for d in res.diagnostics if d.severity == "error"]
        warnings = [
            d.message for d in res.diagnostics if d.severity == "warning"
        ]
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

    def browse(self, file: str, clear: bool = False):
        """
        Add browsable trace information to a command file's outcome if a
        `raw_trace` field is found.

        Arguments:
            file: Path to the command file to update.
            clear: If `True`, clear the browsable trace instead of
                adding it (convenience shortcut for `clear_browsable`
                method).
        """

        if clear:
            self.clear_browsable(file)
            return

        # TODO: introduce cleaner ways to load information from command files

        import delphyne.analysis as analysis
        from delphyne.scripts.command_utils import update_command_file_outcome

        file_path = Path(file)
        workspace_dir = self._workspace_dir_for(file_path)
        config = load_execution_context(workspace_dir, local=file_path)

        with open(file_path, "r") as f:
            content = f.read()

        # Load the tree root from the header
        file_data = yaml.safe_load(content)
        strategy_name = file_data["args"]["strategy"]
        strategy_args = file_data["args"]["args"]
        loader = config.object_loader(extra_objects=STD_COMMANDS)
        strategy = loader.load_strategy_instance(strategy_name, strategy_args)
        root = dp.reify(strategy)

        def add_browsable_trace(outcome_data: Any) -> Any:
            if outcome_data is None:
                return outcome_data
            result = outcome_data.get("result")
            if result is None:
                return outcome_data
            raw_trace = result.get("raw_trace")
            if raw_trace is None:
                print(
                    "No raw_trace found in command outcome.", file=sys.stderr
                )
                return outcome_data
            trace = ty.pydantic_load(dp.ExportableTrace, raw_trace)
            loaded_trace = dp.Trace.load(trace)

            btrace = analysis.compute_browsable_trace(loaded_trace, root=root)
            result["browsable_trace"] = ty.pydantic_dump(fb.Trace, btrace)
            return outcome_data

        new_content = update_command_file_outcome(content, add_browsable_trace)
        with open(file_path, "w") as f:
            f.write(new_content)

    def clear_browsable(self, file: str):
        """
        Clear the browsable trace from a command file's outcome if a
        `browsable_trace` field is found.
        """
        from delphyne.scripts.command_utils import update_command_file_outcome

        file_path = Path(file)
        with open(file_path, "r") as f:
            content = f.read()

        def remove_browsable_trace(outcome_data: Any) -> Any:
            if outcome_data is None:
                return outcome_data
            result = outcome_data.get("result")
            if result is None:
                return outcome_data
            if "browsable_trace" in result:
                del result["browsable_trace"]
            return outcome_data

        new_content = update_command_file_outcome(
            content, remove_browsable_trace
        )
        with open(file_path, "w") as f:
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
