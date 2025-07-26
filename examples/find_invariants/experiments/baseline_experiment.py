from pathlib import Path

import code2inv
from code2inv_experiments import (
    BaselineConfig,
    make_code2inv_baseline_experiment,
)

SMALL = ["gpt-4o-mini"]
LARGE = ["gpt-4o", "o3"]

configs = [
    BaselineConfig(
        bench_name=bench_name,
        model_name=model,
        temperature=temperature if model != "o3" else 1.0,
        max_feedback_cycles=max_feedback_cycles,
        max_dollar_budget=0.2,
        loop=True,
        seed=seed,
    )
    for bench_name in code2inv.load_all_benchmarks()
    for model in [*SMALL, *LARGE]
    for temperature in ([0.7] if model in LARGE else [0.7, 1, 1.5])
    for max_feedback_cycles in ([0, 1, 3] if model in LARGE else [3])
    for seed in range(3)
]

if __name__ == "__main__":
    exp_name = Path(__file__).stem
    exp = make_code2inv_baseline_experiment(exp_name, configs)
    exp.run_cli()
