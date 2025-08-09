# Getting Started

<!-- We do not show the page title -->
<!-- Temporary hack: see https://github.com/squidfunk/mkdocs-material/issues/2163#issuecomment-3156700587 -->
<!-- The following simpler alternative does not work: -->
<!-- <style> .md-content .md-typeset h1 { display: none; } </style> -->
<style>
  /* Hide the h1 title */
  .md-content .md-typeset h1 { display: none; }

  /*
    * By hiding the h1 title, mkdocs will see it as active (scrolled at it)
    * which affects the header. This isn't what we want, as it makes the mike
    * version selector disappear, and it also just isn't what we'd want anyways.
    * Force the main header to be visible and hide the active one.
    */
  .md-header__title--active .md-header__topic {
      opacity: unset;
      pointer-events: unset;
      transform: unset;
      z-index: unset;
  }
  .md-header__title--active .md-header__topic + .md-header__topic {
      opacity: 0;
      pointer-events: unset;
      transform: unset;
      z-index: unset;
  }
</style>

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

```
git clone git@github.com:jonathan-laurent/delphyne.git
cd delphyne
```

Then, to install the Delphyne library in your current Python environment:

```
pip install -e .
```

Note that Python 3.12 (or more recent) is required. Once this is done, it should be possible to run `import delphyne` inside a Python interpreter. Next, you should build the Delphyne vscode extension. For this, assuming you have [Node.js](https://nodejs.org/en/download) installed (version 22 or later), run:

```
cd vscode-ui
npm install
npx vsce package
```

The last command should create a `delphyne-xxx.vsix` extensions archive, which can be installed in vscode using the `Extensions: Install from VSIX` command (use `Ctrl+Shift+P` to search for this command).

### Testing your installation

To test your installation, open VSCode and set the `examples/find_invariants` folder as your workspace root. Click on the Delphyne logo on the Activity Bar to start the Delphyne extension, and open the demonstration file `abduct_and_branch.demo.yaml`. Then, open the command palette (`Ctrl+Shift+P`) and run the command `Delphyne: Evaluate All Demonstrations in File`. Diagnostics should then appear to indicate that all tests passed (but no warning or error). Note that adding new demonstrations requires installing `why3py`, as explained in the example's README.