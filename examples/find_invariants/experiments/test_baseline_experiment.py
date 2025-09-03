import code2inv_experiments as c2i

configs = [
    c2i.BaselineConfig(
        bench_name="1",
        model_name=model,
        temperature=1,
        max_feedback_cycles=3,
        loop=False,
        seed=0,
    )
    for model in ["mistral-small-2503"]
]

if __name__ == "__main__":
    c2i.make_experiment(
        c2i.baseline_experiment, configs, "test-output", __file__
    ).run_cli()
