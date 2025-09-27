<p align="center">
  <a href="https://jonathan-laurent.github.io/delphyne/latest#gh-light-mode-only">
    <img src="docs/assets/logos/black/banner.png#gh-light-mode-only" alt="Delphyne Logo" width="50%"/>
  </a>
  <a href="https://jonathan-laurent.github.io/delphyne/latest#gh-dark-mode-only">
    <img src="docs/assets/logos/white/banner.png#gh-dark-mode-only" alt="Delphyne Logo" width="50%"/>
  </a>
</p>

Delphyne is a programming framework for building _reliable_ and _modular_ LLM applications. It offers a powerful approach for integrating traditional programming and prompting.

Users can describe high-level problem solving _strategies_ as arbitrary programs with unresolved choice points. _Policies_ for navigating the resulting search spaces with LLMs can be independently assembled and tuned. Examples of correct choices can be expressed in a dedicated _demonstration language_ that supports interactive, test-driven development.

## Quick Example

Let us illustrate Delphyne with a complete example. Consider the task of  finding a parameter value that makes a mathematical expression nonnegative for all values of `x`. For example, given expression `x² - 2x + n`, `n = -1` is an incorrect answer (take `x = 0`), but `n = 1` is a correct answer since `x² - 2x + 1 = (x - 1)²`. Here is a Delphyne **strategy** for solving this problem:

```py
import sympy as sp
import delphyne as dp 
from delphyne import Branch, Fail, IPDict, Strategy, strategy

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
        yield from dp.fail("sympy_error", message=str(e))
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
budget = dp.BudgetLimit({dp.NUM_REQUESTS: 8, dp.DOLLAR_PRICE: 1e-4})
res, _ = (find_param_value("2*x**2 - 4*x + n")
          .run_toplevel(dp.PolicyEnv(), serial_policy())
          .collect(budget=budget, num_generated=1))
print(res[0].tracked.value)  # e.g. 2
```

Here, choice points are resolved via zero-shot prompting. To increase reliability, one can provide examples of correct decisions by adding **demonstrations**. Demonstrations can be expressed in a dedicated language that supports an interactive, test-driven development workflow. In the screenshot below, the VSCode extension indicates the next unresolved choice point for solving a particular problem instance, which the user can answer before receiving new feedback:

![](docs/assets/screenshot/readme-extension-example/dark.png#gh-dark-mode-only)
![](docs/assets/screenshot/readme-extension-example/light.png#gh-light-mode-only)

A complete explanation of this example is provided in the [Delphyne Manual](https://jonathan-laurent.github.io/delphyne/latest/manual/overview/).

## Features Overview

- **Separate your business logic from your search logic.** Prompting offers a powerful but unreliable programming primitive, making validation and search essential. At the same time, retries and backtracking are costly. Resolving this tension requires careful tuning. Delphyne enforces a strict separation between _business logic_ (i.e. strategies) and _search logic_ (i.e. policies), allowing users to iterate on the latter without any refactoring.
- **Write and repair demonstrations interactively.** Few-shot examples are crucial components of an LLM pipeline, but writing them and keeping them consistent as the pipeline evolves can be error-prone and time-consuming. Delphyne's dedicated [_demonstration language_](https://jonathan-laurent.github.io/delphyne/latest/manual/demos/) allows bundling examples with unit tests and updating them interactively.
- **Refactor without fear.** Delphyne leverages _static types_ to enforce consistency between strategies and policies, and a _testing DSL_ to enforce consistency between strategies and demonstrations. Modifying policies is _guaranteed_ to never break demonstrations.
- **Compose heterogeneous strategies in a modular fashion.** Delphyne allows modularly composing heterogeneous strategies while keeping the associated policies independent. This is enabled by having arbitrary search algorithms cooperate through a [shared protocol](https://jonathan-laurent.github.io/delphyne/latest/reference/policies/streams/). In addition, any prompt can be _locally_ and _transparently_ refined into a dedicated strategy.
- **Leverage advanced search algorithms.** Thanks to an [extensible effect system](https://jonathan-laurent.github.io/delphyne/latest/manual/strategies/#adding-effects), strategies can produce search trees with arbitrary metadata, structure, and shape, which advanced algorithms (e.g. [HyperTree Proof Search](https://arxiv.org/pdf/2205.11491)) can exploit. Tree transformers enable conversions between tree types, allowing the same strategy to pair with vastly different algorithms.
- **Precisely control your resource spending.** Delphyne offers a resource-aware, combinator-based [language](https://jonathan-laurent.github.io/delphyne/latest/manual/policies/) for defining search policies that prevents overspending _by construction_.
- **Leverage the full Python ecosystem.** Delphyne integrates seamlessly with Python libraries and tooling. It is fully typed and works smoothly with the Python debugger.
- **Access a rich library of components.** Delphyne comes with batteries included: its standard library supports many models and prompting idioms, including structured output, tool use, classification, assistant priming, and chain of thought.
- **Run reproducible experiments.** Delphyne provides utilities for benchmarking and tuning LLM pipelines. Request caching allows experiments to be replicated at no cost.

## Installation

First, download the Delphyne repository and enter it:

```sh
git clone git@github.com:jonathan-laurent/delphyne.git
cd delphyne
git checkout v0.13.0  # latest stable version
```

Then, to install the Delphyne library and CLI in your current Python environment:

```sh
pip install -e ".[dev]"
```

Note that Python 3.12 (or more recent) is required, since Delphyne makes heavy use of [PEP 695 generics](https://peps.python.org/pep-0695/). Next, you should build the Delphyne vscode extension. For this, assuming you have [Node.js](https://nodejs.org/en/download) installed (version 22 or later), run:

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

To test the Delphyne extension, we recommend reading the corresponding [manual chapter](https://jonathan-laurent.github.io/delphyne/latest/manual/extension/), opening the `examples/find_invariants/abduct_and_branch.demo.yaml` demonstration file, and then executing the `Delphyne: Evaluate All Demonstrations in File` command (accessible from the palette via `Ctrl+Shift+P`). Diagnostics should appear to indicate that all tests passed.

## Learning More

- The key concepts underlying Delphyne are presented in the [Manual](https://jonathan-laurent.github.io/delphyne/latest/manual/overview/).
- A complete [API Reference](https://jonathan-laurent.github.io/delphyne/latest/reference/strategies/trees/) is also available, along with a series of [How-To Guides](https://jonathan-laurent.github.io/delphyne/latest/how-to-guides/).
- The `examples` directory provides a gallery of Delphyne programs (see  `README` files).
- For a more technical presentation, see the original paper that introduces Delphyne: [_Oracular Programming: A Modular Foundation for Building LLM-Enabled Software_](https://arxiv.org/abs/2502.05310).

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
