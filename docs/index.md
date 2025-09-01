# Getting Started

<!-- We do not show the page title -->
<style> .md-content .md-typeset h1 { display: none; } </style>

<p align="center" style="margin-bottom: 45px">
  <img src="assets/logos/black/banner.png#only-light" alt="Delphyne Logo" width="50%"/>
  <img src="assets/logos/white/banner.png#only-dark" alt="Delphyne Logo" width="50%"/>
</p>

Delphyne is a programming framework for building _reliable_ and _modular_ LLM applications. It is based on a new paradigm named _oracular programming_, where high-level problem-solving strategies are expressed as nondeterministic programs whose choice points are annotated with examples and resolved by LLMs. Delphyne combines three languages:

- A _strategy language_ embedded in Python that allows writing nondeterministic programs that can be reified into (modular) search trees.
- A _policy language_ for specifying ways to navigate such trees (with LLM guidance) by composing reusable search primitives.
- A _demonstration language_ for describing successful _and_ unsuccessful search scenarios to be used as training or prompting examples. A dedicated language server allows writing demonstrations interactively and keeping them synchronized with evolving strategies.

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

<!-- ## Learning More -->

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