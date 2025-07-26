from pathlib import Path

import code2inv
from code2inv_experiments import (
    AbductionConfig,
    make_code2inv_abduction_experiment,
    run_app,
)

configs = [
    AbductionConfig(
        bench_name=bench,
        model_cycle=[("gpt-4o-mini", 1)],
        temperature=temp,
        num_concurrent=num_concurrent,
        max_requests_per_attempt=max_requests_per_attempt,
        max_dollar_budget=0.2,
        seed=seed,
    )
    for bench in code2inv.load_all_benchmarks()
    for seed in range(3)
    for num_concurrent in [8]
    for temp in [1.5, 1.7]
    for max_requests_per_attempt in [4, 8]
]

if __name__ == "__main__":
    exp_name = Path(__file__).stem
    exp = make_code2inv_abduction_experiment(exp_name, configs)
    run_app(exp)
