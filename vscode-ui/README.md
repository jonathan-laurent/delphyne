# VSCode Extension for Delphyne

## Installation

To install all dependencies:

```sh
npm install
npm install -g @vscode/vsce
```

Then, start the extension in debugging mode using `F5`.

Somehow, `npm install` requires a Python installation because of  `node-gyp`.

To install the vscode extension:

```sh
npm install -g vsce
vsce package
code --install-extension *.vsix
```

To typecheck everything:

```sh
npx tsc --noEmit
```

Instead of the last step, one can go to the "Extensions" menus, click on the three dots and select "Install from VSIX".


## References

- https://github.com/microsoft/vscode-extension-samples/tree/main/decorator-sample
- https://stackoverflow.com/questions/48723181/can-a-language-in-visual-studio-code-be-extended
- https://github.com/microsoft/vscode-extension-samples/tree/main/codelens-sample
- https://github.com/microsoft/vscode-extension-samples/tree/main/tree-view-sample
- https://github.com/microsoft/vscode-extension-samples/tree/main/decorator-sample
- Obtain breadcrumbs for YAML:
  - https://stackoverflow.com/questions/59653826/how-to-access-the-text-of-vscode-breadcrumb-from-extension
  - https://stackoverflow.com/questions/59653826/how-to-access-the-text-of-vscode-breadcrumb-from-extension
  - https://github.com/cunneen/vscode-copy-breadcrumbs/blob/main/src/extension.ts


### VSCode API

- [Diagnostics info](https://code.visualstudio.com/api/language-extensions/programmatic-language-features#provide-diagnostics)
- [View actions](https://code.visualstudio.com/api/extension-guides/tree-view#view-actions)


### YAML

- [Stringify options](https://eemeli.org/yaml/#tostring-options)

### Validation and schema support in Typescript

The auto-conversion tools are imperfect. pydantic-to-typescript raised a runtime error on delphyne.demos and json-schema-to-typescript generates custom types for all fields, which is not ideal...

- [YAML library documentation](https://eemeli.org/yaml/#yaml)
- [Transform schemas into types via type computations](https://github.com/ThomasAribart/json-schema-to-ts#readme)
- [Compile schemas into type definitions](https://github.com/bcherny/json-schema-to-typescript#readme)
- [Compile Pydantic to Typescript](https://github.com/phillipdupuis/pydantic-to-typescript)
- [Most popular JSON validator for JS](https://github.com/ajv-validator/ajv)
- [Other JSON schema validator](https://github.com/tdegrunt/jsonschema)


### Tips

See what processes listen to port 8000 and kill them.

```sh
sudo lsof -t -i :8000
sudo kill -9 $(sudo lsof -t -i :8000)
```

If you are bothered by copilot actions:

```json
"github.copilot.editor.enableCodeActions": false
```

## Useful VSCode Shortcuts

- Close a tab: `Cmd+W` (and then `Cmd+D` to not save)
- Select tab groups: `Cmd+{1,2}`
- Go to line: `Ctrl+G`
- Expand AST selection: `Cmt+Ctrl+Shift+->`