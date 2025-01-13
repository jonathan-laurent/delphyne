<!-- We do not show the page title -->
<style> .md-content .md-typeset h1 { display: none; } </style>
<p align="center">
  <img src="assets/logos/black/banner.png#only-light" alt="Delphyne Logo" width="50%"/>
  <img src="assets/logos/white/banner.png#only-dark" alt="Delphyne Logo" width="50%"/>
</p>


Delphyne is a framework for solving problems.


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

Note that Python 3.12 (or more recent) is required. Once this is done, it should be possible to run `import delphyne` inside a Python interpreter. Next, you should build the Delphyne vscode extension. For this, assuming you have [Node.js](https://nodejs.org/en/download) installed (version 22 or later), run:

```
cd vscode-ui
npm install
npx vsce package
```

The last command should create a `delphyne-xxx.vsix` extensions archive, which can be installed in vscode using the `Extensions: Install from VSIX` command (use `Ctrl+Shift+P` to search for this command).