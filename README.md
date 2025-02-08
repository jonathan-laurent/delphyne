<p align="center">
  <img src="docs/assets/logos/black/banner.png#gh-light-mode-only" alt="Delphyne Logo" width="50%"/>
  <img src="docs/assets/logos/white/banner.png#gh-dark-mode-only" alt="Delphyne Logo" width="50%"/>
</p>


Delphyne is a programming framework for building _reliable_ and _modular_ LLM applications. It is based on a new paradigm named _oracular programming_, where high-level problem-solving strategies are expressed as nondeterministic programs whose choice points are annotated with examples and resolved by LLMs. Delphyne combines three languages:

- A _strategy language_ embedded in Python that allows writing nondeterministic programs that can be reified into (modular) search trees.
- A _policy language_ for specifying ways to navigate such trees (with LLM guidance) by composing reusable search primitives.
- A _demonstration language_ for describing successful _and_ unsuccessful search scenarios to be used as training or prompting examples. A dedicated language server allows writing demonstrations interactively and keeping them synchronized with evolving strategies.

> [!WARNING]
> Delphyne is still under development and is evolving quickly. You should expect some rough edges.


## Installation

First, download the Delphyne repository and enter it:

```
git clone git@github.com:jonathan-laurent/delphyne.git
cd delphyne
```

Then, to install the Delphyne library in your current Python environment:

```
pip install -e .
```

Note that Python 3.12 (or more recent) is required. Once this is done, it should be possible to run `import delphyne` inside a Python interpreter. Next, you should build the Delphyne vscode extension. For this, assuming you have [Node.js](https://nodejs.org/en/download) installed, run:

```
cd vscode-ui
npm install
npx vsce package
```

The last command should create a `delphyne-xxx.vsix` extensions archive, which can be installed in vscode using the `Extensions: Install from VSIX` command (use `Ctrl+Shift+P` to search for this command).

### Testing your installation

To test your installation, open VSCode and set the `examples/find_invariants` folder as your workspace root. Click on the Delphyne logo on the Activity Bar to start the Delphyne extension, and open the demonstration file `find_invariants.demo.yaml`. Then, open the command palette (`Ctrl+Shift+P`) and run the command `Delphyne: Evaluate All Demonstrations in File`. Diagnostics should then appear to indicate that all tests passed (but no warning or error). Note that adding new demonstrations requires installing `why3py`, as explained in the example's README.


## Getting Started

To learn about the core concepts underlying Delphyne, we recommend that you read the paper: [_Oracular Programming: A Modular Foundation for Building LLM-Enabled Software_](https://arxiv.org/). An easier introduction is coming soon.

You can then look at the `examples/find_invariants` folder for an example of an oracular program. 