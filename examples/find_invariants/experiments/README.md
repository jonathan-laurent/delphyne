# Experiments

This directory contains the source and data for the invariant synthesis experiments from the [original paper on _oracular programming_](https://arxiv.org/abs/2502.05310). It also contains a tiny variant of these experiments (`test_*` files), to be used as a part of the Delphyne's extended test suite.

The experiment output directory `output` is not included in the versioned repository for reasons of size. To reproduce the experiment using the same LLM request cache, you can paste it in this folder and then run:

```py
python abduction_experiment.py replay
python baseline_experiment.py replay
```

Generated experiment summaries were copied in the `analysis/data` folder and should coincide with the content of `output`. To reproduce the paper's main table from them, run:

```py
python analysis/analysis.py
```

## Experiment info

- Experiment date: 09/03/2025
- Delphyne version: 0.9.1
- Alt-Ergo version: 2.6.2
- Why3 version: 1.8.1
- Model snapshots:
  - [gpt-4o](https://platform.openai.com/docs/models/gpt-4o): `gpt-4o-2024-08-06`
  - [gpt-4o-mini](https://platform.openai.com/docs/models/gpt-4o-mini) `gpt-4o-mini-2024-07-18`
  - [o3](https://platform.openai.com/docs/models/o3): `o3-2025-04-16`