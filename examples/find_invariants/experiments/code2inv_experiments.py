"""
Code2Inv Experiments
"""

from collections.abc import Sequence
from dataclasses import dataclass

import code2inv
import delphyne as dp
import delphyne.stdlib.commands as cmd

BENCHS = code2inv.load_all_benchmarks()


@dataclass
class AbductionConfig:
    bench_name: str
    model_cycle: Sequence[tuple[str, int]]
    temperature: float
    num_completions: int
    max_requests_per_attempt: int
    max_dollar_budget: float
    seed: int

    def instantiate(self, context: object):
        return cmd.RunStrategyArgs(
            strategy="prove_program_by_recursive_abduction",
            args={"prog": BENCHS[self.bench_name]},
            policy="prove_program_by_saturation",
            policy_args={
                "model_cycle": self.model_cycle,
                "temperature": self.temperature,
                "num_completions": self.num_completions,
                "max_requests_per_attempt": self.max_requests_per_attempt,
            },
            budget={dp.DOLLAR_PRICE: self.max_dollar_budget},
            log_long_computations=("info", 1.0),
        )


@dataclass
class BaselineConfig:
    bench_name: str
    model_name: str
    temperature: float
    max_feedback_cycles: int
    seed: int
    loop: bool = False
    max_dollar_budget: float | None = 0.2

    def instantiate(self, context: object):
        budget: dict[str, float] = {}
        if self.max_dollar_budget is not None:
            budget[dp.DOLLAR_PRICE] = self.max_dollar_budget
        return cmd.RunStrategyArgs(
            strategy="prove_program_interactive",
            args={"prog": BENCHS[self.bench_name]},
            policy="prove_program_interactive_policy",
            policy_args={
                "model_name": self.model_name,
                "temperature": self.temperature,
                "max_feedback_cycles": self.max_feedback_cycles,
                "loop": self.loop,
            },
            budget=budget,
        )
