from pathlib import Path

import code2inv_experiments as c2i
from delphyne.stdlib.experiments.experiment_launcher import quick_experiment

configs = [
    c2i.BaselineConfig(
        bench_name="1",
        model_name=model,
        temperature=1,
        max_feedback_cycles=3,
        seed=0,
    )
    for model in ["mistral-small-2503"]
]

if __name__ == "__main__":
    quick_experiment(
        c2i.baseline_experiment,
        configs,
        name=Path(__file__).stem,
        workspace_root=Path(__file__).parent.parent,
        modules=c2i.MODULES,
        demo_files=c2i.DEMO_FILES,
        output_dir=Path("experiments") / "test-output",
    ).run_cli()
