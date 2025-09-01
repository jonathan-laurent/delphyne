# Getting Started

<!-- We do not show the page title -->
<style> .md-content .md-typeset h1 { display: none; } </style>

<p align="center" style="margin-bottom: 45px">
  <img src="assets/logos/black/banner.png#only-light" alt="Delphyne Logo" width="50%"/>
  <img src="assets/logos/white/banner.png#only-dark" alt="Delphyne Logo" width="50%"/>
</p>

Delphyne is a programming framework for building _reliable_ and _modular_ LLM applications. It offers a powerful approach for integrating traditional programming and prompting.

Users can describe high-level problem solving _strategies_ as arbitrary programs with unresolved choice points. _Policies_ for navigating the resulting search spaces with LLMs can be independently assembled and tuned. Examples of correct choices can be expressed in a dedicated _demonstration language_ that supports interactive test-driven developement.

<!-- and used for few-shot prompting. -->
<!-- by composing reusable components -->

<!-- Delphyne is a programming framework for building _reliable_ and _modular_ LLM applications. It is based on a new paradigm named _oracular programming_, where high-level problem-solving strategies are expressed as nondeterministic programs whose choice points are annotated with examples and resolved by LLMs. Delphyne combines three languages:

- A _strategy language_ embedded in Python that allows writing nondeterministic programs that can be reified into (modular) search trees.
- A _policy language_ for specifying ways to navigate such trees (with LLM guidance) by composing reusable search primitives.
- A _demonstration language_ for describing successful _and_ unsuccessful search scenarios to be used as training or prompting examples. A dedicated language server allows writing demonstrations interactively and keeping them synchronized with evolving strategies. -->

## Quick Example

Let us illustrate Delphyne with a complete example. Consider the task of  finding a parameter value that makes a mathematical expression nonnegative for all values of `x`. For example, given expression `x² - 2x + n`, `n = -1` is an incorrect answer (take `x = 0`), but `n = 1` is a correct answer since `x² - 2x + 1 = (x - 1)²`. Here is a Delphyne **strategy** for solving this problem:

```py
import sympy as sp
from typing import assert_never
import delphyne as dp 
from delphyne import Branch, Fail, Strategy, strategy

@strategy
def find_param_value(expr: str) -> Strategy[Branch | Fail, IPDict, int]:
    """
    Find an integer `n` that makes a given math expression nonnegative
    for all real `x`. Prove that the resulting expression is nonnegative
    by rewriting it into an equivalent form.
    """
    x, n = sp.Symbol("x", real=True), sp.Symbol("n")
    symbs = {"x": x, "n": n}
    try:
        n_val = yield from dp.guess(int, using=[expr])
        expr_sp = sp.parse_expr(expr, symbs).subs({n: n_val})
        equiv = yield from dp.guess(str, using=[str(expr_sp)])
        equiv_sp = sp.parse_expr(equiv, symbs)
        equivalent = (expr_sp - equiv_sp).simplify() == 0
        yield from dp.ensure(equivalent, "not_equivalent")
        yield from dp.ensure(equiv_sp.is_nonnegative, "not_nonneg")
        return n_val
    except Exception as e:
        assert_never((yield from dp.fail("sympy_error", message=str(e))))
```

A strategy is a program with unresolved choice points, represented here by the `guess` operator. At runtime, LLM oracles are tasked with producing return values for `guess`, in such a way as to pass all `ensure` assertions. Such a program induces a search tree that can be explored in many different ways, as specified by a separate **policy**:

```py
def serial_policy():
    model = dp.standard_model("gpt-5-mini")
    return dp.dfs() & {
        "n_val": dp.few_shot(model),
        "equiv": dp.take(2) @ dp.few_shot(model)}
```

This policy uses _sequential depth-first search_ (`dfs`), making at most two proof attempts for every parameter guess and sampling choices using `gpt-5-mini`. Here is another policy that uses a parallel variant of `dfs`, where multiple completions are repeatedly sampled and the resulting branches explored in parallel:

```py
def parallel_policy():
    model = dp.standard_model("gpt-5-mini")
    return dp.loop() @ dp.par_dfs() & {
        "n_val": dp.few_shot(model, max_requests=1, num_completions=3),
        "equiv": dp.few_shot(model, max_requests=1, num_completions=2)}
```

Given an associated policy and a _budget limit_, our strategy can now be executed:

```py
budget = dp.BudgetLimit({dp.NUM_REQUESTS: 8, dp.DOLLAR_PRICE: 1e-3})
res, _ = (find_param_value("2*x**2 - 4*x + n")
          .run_toplevel(dp.PolicyEnv(), serial_policy())
          .collect(budget=budget, num_generated=1))
print(res[0].tracked.value)  # e.g. 2
```

Here, choice points are resolved via zero-shot prompting. To increase reliability, one can provide examples of correct decisions by adding **demonstrations**. Demonstrations can be expressed in a dedicated language that supports an interactive, test-driven development workflow. In the screenshot below, the VSCode extension indicates the next unresolved choice point for solving a particular problem instance, which the user can answer before receiving new feedback:

![](./assets/screenshot/readme-extension-example/dark.png#only-dark)
![](./assets/screenshot/readme-extension-example/light.png#only-light)

A complete explanation of this example is provided in the [Delphyne Manual](./manual/overview.md).

## Features Overview

## Installation {#installation}

First, download the Delphyne repository and enter it:

```sh
git clone git@github.com:jonathan-laurent/delphyne.git
cd delphyne
git checkout v0.7.0  # latest stable version
```

Then, to install the Delphyne library and CLI in your current Python environment:

```sh
pip install -e ".[dev]"
```

Note that Python 3.12 (or more recent) is required. Next, you should build the Delphyne vscode extension. For this, assuming you have [Node.js](https://nodejs.org/en/download) installed (version 22 or later), run:

```
cd vscode-ui
npm install
npx vsce package
```

The last command should create a `delphyne-xxx.vsix` extensions archive, which can be installed in vscode using the `Extensions: Install from VSIX` command (use `Ctrl+Shift+P` to search for it in the command palette).

### Testing your installation

You can verify your installation by running the short test suite:

```
make test
```

To test the Delphyne extension, we recommend reading the corresponding [manual chapter](./manual/extension.md), opening the `examples/find_invariants/abduct_and_branch.demo.yaml` demonstration file, and then executing the `Delphyne: Evaluate All Demonstrations in File` command (accessible from the palette via `Ctrl+Shift+P`). Diagnostics should appear to indicate that all tests passed.

## Learning More

## Citing this Work

If you use Delphyne in an academic paper, please cite our work as follows:

```bib
@article{oracular-programming-2025,
  title={Oracular Programming: A Modular Foundation for Building LLM-Enabled Software},
  author={Laurent, Jonathan and Platzer, Andr{\'e}},
  journal={arXiv preprint arXiv:2502.05310},
  year={2025}
}
```

## Acknowledgements

This work was supported by the [Alexander von Humboldt Professorship program](https://www.humboldt-foundation.de/en/apply/sponsorship-programmes/alexander-von-humboldt-professorship).