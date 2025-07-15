import code2inv
from code2inv_experiments import (EnsembleConfig,
                                  make_code2inv_ensemble_experiment, run_app)

configs = [
    EnsembleConfig(
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
    exp = make_code2inv_ensemble_experiment("saturation_final_tuned", configs)
    run_app(exp)
