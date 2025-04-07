"""
Standard Command Line Tools for Delphyne
"""

from collections.abc import Sequence
from pathlib import Path

import fire  # type: ignore


class DelphyneApp:
    """
    Delphyne Command Line Application
    """

    def __init__(self, strategy_dirs: Sequence[Path], modules: Sequence[str]):
        self.strategy_dirs = strategy_dirs
        self.modules = modules

    def check_demo(
        self,
        file: Path,
        ensure_no_error: bool = False,
        ensure_no_warning: bool = False,
    ):
        """
        Check a demo file.
        """
        from delphyne.scripts.demonstrations import check_demo_file

        feedback = check_demo_file(file, self.strategy_dirs, self.modules)
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


if __name__ == "__main__":
    fire.Fire(DelphyneApp)  # type: ignore
