"""
Scripts for manipulating demonstrations
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from delphyne import analysis, stdlib
from delphyne import core as dp
from delphyne.utils import typing as ty


@dataclass
class DemoFileFeedback:
    errors: list[str]
    warnings: list[str]

    def add_diagnostic(
        self, demo_name: str, loc: str, diag: analysis.Diagnostic
    ):
        t, s = diag
        pp = f"[{demo_name}:{loc}] {s}"
        if t == "error":
            self.errors.append(pp)
        elif t == "warning":
            self.warnings.append(pp)

    def add_diagnostics(self, demo_name: str, f: analysis.DemoFeedback):
        if isinstance(f, analysis.QueryDemoFeedback):
            for i, d in enumerate(f.diagnostics):
                self.add_diagnostic(demo_name, str(i), d)
            for i, d in enumerate(f.answer_diagnostics):
                self.add_diagnostic(demo_name, str(i), d[1])
        else:
            for i, d in enumerate(f.global_diagnostics):
                self.add_diagnostic(demo_name, str(i), d)
            for i, d in enumerate(f.query_diagnostics):
                self.add_diagnostic(demo_name, str(i), d[1])
            for i, d in enumerate(f.answer_diagnostics):
                self.add_diagnostic(demo_name, str(i), d[1])
            for i, d in enumerate(f.implicit_answers):
                msg = f"Implicit answer for {d.query_name}({d.query_args})"
                self.add_diagnostic(demo_name, str(i), ("warning", msg))


def check_demo_file(
    file: Path, strategy_dirs: Sequence[Path], modules: Sequence[str]
) -> DemoFileFeedback:
    """
    TODO: we should better report line numbers.
    """
    context = analysis.DemoExecutionContext(strategy_dirs, modules)
    demos_json = yaml.safe_load(open(file, "r").read())
    demos = ty.pydantic_load(list[dp.Demo], demos_json)
    extra = stdlib.stdlib_globals()
    ret = DemoFileFeedback([], [])
    for i, d in enumerate(demos):
        feedback = analysis.evaluate_demo(d, context, extra)
        name = d.demonstration if d.demonstration else f"#{i}"
        ret.add_diagnostics(name, feedback)
    return ret
