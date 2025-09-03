import code2inv_experiments as c2i

configs = [
    c2i.AbductionConfig(
        bench_name=bench,
        model_cycle=[("gpt-4o-mini", 1)],
        temperature=temp,
        num_completions=num_completions,
        max_requests_per_attempt=max_requests_per_attempt,
        max_dollar_budget=0.2,
        seed=seed,
    )
    for bench in c2i.BENCHS
    for seed in range(3)
    for num_completions in [8]
    for temp in [1.5]
    for max_requests_per_attempt in [4, 8]
]

if __name__ == "__main__":
    c2i.make_experiment(
        c2i.abduction_experiment, configs, "output", __file__
    ).run_cli()
