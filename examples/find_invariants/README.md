# Generating Loop Invariants with Delphyne and Why3

This example project demonstrates four different strategies for finding loop invariants for single-loop, imperative Why3 programs. The first two are featured in the [original paper on _oracular programming_](https://arxiv.org/abs/2502.05310).

- `baseline.py`: a simple conversational agent baseline based on the standard  `interact` strategy.
- `abduct_and_saturate.py`: a strategy based on iterative abduction, which lev erages the `Abduction` effect from the standard library. As demonstrated in the aforementioned paper, it can solve all problems in the Code2Inv benchmark suite, at a cost 9x to 27x lower depending on the LLM being used.
- `abduct_and_branch.py`: a strategy that also uses abduction, in combination with branching and value nodes. It is associated a policy that uses the `bestfs` search algorithm. This strategy hasn't been tuned and does not currently perform particularly well, but it provides an example of using value nodes, providing demonstrations involving negative examples and defining advanced policies.
- `one_guess_baseline.py`: an even simpler baseline, which simply asks an LLM to guess invariants and checks the results, without providing repair opportunities.

## Getting Started

Some demonstration files such as `abduct_and_branch.demo.yaml`, in which the result of solver calls is hardcoded, can be evaluated (`make test`) without installing Why3 or Why3py. All test experiments can be similarly replicated (`make test-exp`), and command files rerun (`make execute-commands`), since a request cache is available for them. However, to add new demonstrations or run new commands or experiments, `why3py` must be installed. Instructions are available in `examples/libraries/why3py/README.md`.