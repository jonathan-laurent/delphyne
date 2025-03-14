# VSCode Extension for Delphyne

## Installation

To install all dependencies:

```sh
npm install
```

Then, start the extension in debugging mode using `F5`.

Somehow, `npm install` requires a Python installation because of  `node-gyp`.

To install the vscode extension:

```sh
npm install @vscode/vsce
npx vsce package
code --install-extension *.vsix
```

To typecheck everything:

```sh
npx tsc --noEmit
```

Instead of the last step, one can go to the "Extensions" menus, click on the three dots and select "Install from VSIX".

To make the most out of the extension, we recommend also installing the YAML extension.