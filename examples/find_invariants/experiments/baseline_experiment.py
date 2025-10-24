import code2inv_experiments as c2i

O3 = "o3-2025-04-16"
SMALL = ["gpt-4o-mini-2024-07-18"]
LARGE = ["gpt-4o-2024-08-06", O3]

configs = [
    c2i.BaselineConfig(
        bench_name=bench_name,
        model_name=model,
        temperature=temperature if model != O3 else 1.0,
        max_feedback_cycles=max_feedback_cycles,
        max_dollar_budget=0.2,
        loop=True,
        seed=seed,
    )
    for bench_name in c2i.BENCHS
    for model in [*SMALL, *LARGE]
    for temperature in ([0.7] if model in LARGE else [0.7, 1, 1.5])
    for max_feedback_cycles in ([0, 1, 3] if model in LARGE else [3])
    for seed in range(3)
]

if __name__ == "__main__":
    c2i.make_experiment(
        c2i.BaselineConfig, configs, "output", __file__
    ).run_cli()
