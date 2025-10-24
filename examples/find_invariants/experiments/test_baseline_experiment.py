# pyright: strict

import code2inv_experiments as c2i
import delphyne as dp

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
    dp.Experiment(
        config_class=c2i.BaselineConfig,
        context=dp.workspace_execution_context(__file__),
        configs=configs,
        output_dir=f"experiments/test-output/{dp.path_stem(__file__)}",
    ).run_cli()
