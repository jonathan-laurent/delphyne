from pathlib import Path

import code2inv_experiments as c2i
from delphyne.stdlib.experiments.experiment_launcher import quick_experiment

configs = [
    c2i.AbductionConfig(
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
    quick_experiment(
        c2i.abduction_experiment,
        configs,
        name=Path(__file__).stem,
        workspace_root=Path(__file__).parent.parent,
        modules=c2i.MODULES,
        demo_files=c2i.DEMO_FILES,
        output_dir=Path("experiments") / "test-output",
    ).run_cli()
