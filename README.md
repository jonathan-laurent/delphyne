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