# The Delphyne Framework for Few-Shot Programming

## Install

```sh
pip install -e .
```

To build the documentation:

```sh
pip install mkdocs mkdocstrings[python] mkdocs-autolinks-plugin
mkdocs serve
```

Current pyright version: 1.1.378
To check the pyright version used by pylance, check "Output / Python Language Server" and look for "pyright".

```sh
pip install pyright==1.1.378  # For the one in sync with pylance
pip install pyright -U  # for the latest one
```

## Tricks

To create a test notebook:

```py
%load_ext autoreload
%autoreload 2
from test_strategies import *
```

To count lines in the codebase:

```py
find . -name '*.py' -print0 | xargs -0 wc -l
find vscode-ui/src/ -name '*.ts' -print0 | xargs -0 wc -l

npm install -g cloc
cloc . --exclude-dir=node_modules,out,.vscode-test --include-lang=python,typescript
```