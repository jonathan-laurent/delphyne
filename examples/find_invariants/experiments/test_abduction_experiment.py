import code2inv_experiments as c2i

configs = [
    c2i.AbductionConfig(
        bench_name=bench,
        model_cycle=[("gpt-4.1-nano", 2), ("gpt-4.1-mini", 1)],
        temperature=1.5,
        num_completions=4,
        max_requests_per_attempt=4,
        max_dollar_budget=0.1,
        seed=0,
    )
    for bench in ["1", "7"]
]

if __name__ == "__main__":
    c2i.make_experiment(
        c2i.abduction_experiment, configs, "test-output", __file__
    ).run_cli()
