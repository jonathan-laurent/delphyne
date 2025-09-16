"""
Scripts for manipulating demonstrations
"""

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
            for i, t in enumerate(f.test_feedback):
                for d in t.diagnostics:
                    self.add_diagnostic(demo_name, f"test_{i}", d)
            for d in f.global_diagnostics:
                self.add_diagnostic(demo_name, "global", d)
            for i, d in f.query_diagnostics:
                self.add_diagnostic(demo_name, f"query_{i}", d)
            for (qi, ai), d in f.answer_diagnostics:
                self.add_diagnostic(demo_name, f"query_{qi}:answer_{ai}", d)
            for cat, d in f.implicit_answers.items():
                msg = f"Implicit answers: {cat} ({len(d)} answer(s))"
                self.add_diagnostic(demo_name, "implicit", ("warning", msg))


def check_demo_file(
    file: Path, context: analysis.DemoExecutionContext
) -> DemoFileFeedback:
    # TODO: we should better report line numbers.
    demos_json = yaml.safe_load(open(file, "r").read())
    demos = ty.pydantic_load(list[dp.Demo], demos_json)
    extra = stdlib.stdlib_globals()
    ret = DemoFileFeedback([], [])
    for i, d in enumerate(demos):
        feedback = analysis.evaluate_demo(d, context, extra)
        name = d.demonstration if d.demonstration else f"#{i}"
        ret.add_diagnostics(name, feedback)
    return ret
