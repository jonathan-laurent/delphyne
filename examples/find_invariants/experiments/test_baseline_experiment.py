from pathlib import Path

from code2inv_experiments import (
    BaselineConfig,
    make_code2inv_baseline_experiment,
)

configs = [
    BaselineConfig(
        bench_name="1",
        model_name=model,
        temperature=1,
        max_feedback_cycles=3,
        seed=0,
    )
    for model in [
        "mistral-small-2503"
    ]  # ["mistral-small-2503"]  # ["gpt-4.1-mini", "o3"]
]

if __name__ == "__main__":
    exp_name = Path(__file__).stem
    exp = make_code2inv_baseline_experiment(exp_name, configs)
    exp.run_cli()
