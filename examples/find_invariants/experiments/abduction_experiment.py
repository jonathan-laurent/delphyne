from pathlib import Path

import code2inv_experiments as c2i
from delphyne.stdlib.experiments.experiment_launcher import quick_experiment

configs = [
    c2i.AbductionConfig(
        bench_name=bench,
        model_cycle=[("gpt-4o-mini", 1)],
        temperature=temp,
        num_concurrent=num_concurrent,
        max_requests_per_attempt=max_requests_per_attempt,
        max_dollar_budget=0.2,
        seed=seed,
    )
    for bench in c2i.BENCHS
    for seed in range(3)
    for num_concurrent in [8]
    for temp in [1.5, 1.7]
    for max_requests_per_attempt in [4, 8]
]

if __name__ == "__main__":
    quick_experiment(
        c2i.abduction_experiment,
        configs,
        name=Path(__file__).stem,
        workspace_root=Path(__file__).parent.parent,
        modules=c2i.MODULES,
        demo_files=c2i.DEMO_FILES,
    ).run_cli()
