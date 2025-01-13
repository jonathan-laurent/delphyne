<p align="center">
  <img src="docs/assets/logos/black/banner.png#gh-light-mode-only" alt="Delphyne Logo" width="50%"/>
  <img src="docs/assets/logos/white/banner.png#gh-dark-mode-only" alt="Delphyne Logo" width="50%"/>
</p>


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