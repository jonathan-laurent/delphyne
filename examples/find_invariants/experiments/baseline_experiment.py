from pathlib import Path

import code2inv_experiments as c2i
from delphyne.stdlib.experiments.experiment_launcher import quick_experiment

SMALL = ["gpt-4o-mini"]
LARGE = ["gpt-4o", "o3"]

configs = [
    c2i.BaselineConfig(
        bench_name=bench_name,
        model_name=model,
        temperature=temperature if model != "o3" else 1.0,
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
    quick_experiment(
        c2i.baseline_experiment,
        configs,
        name=Path(__file__).stem,
        workspace_root=Path(__file__).parent.parent,
        modules=c2i.MODULES,
        demo_files=c2i.DEMO_FILES,
    ).run_cli()
