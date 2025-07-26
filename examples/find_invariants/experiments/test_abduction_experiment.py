from pathlib import Path

from code2inv_experiments import (
    AbductionConfig,
    make_code2inv_abduction_experiment,
    run_app,
)

configs = [
    AbductionConfig(
        bench_name=bench,
        model_cycle=[("gpt-4.1-nano", 2), ("gpt-4.1-mini", 1)],
        temperature=1.5,
        num_concurrent=4,
        max_requests_per_attempt=4,
        max_dollar_budget=0.1,
        seed=0,
    )
    for bench in ["1", "7"]
]

if __name__ == "__main__":
    exp_name = Path(__file__).stem
    exp = make_code2inv_abduction_experiment(exp_name, configs)
    run_app(exp)
