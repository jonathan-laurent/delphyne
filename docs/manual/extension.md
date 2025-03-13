# VSCode Extension

A _Visual Studio Code_ extension is available for interactively writing demonstrations, navigating strategy trees, and running oracular programs.

<p align="center">
  <img src="/assets/extension-screenshot.png" alt="Extension Screenshot" style="width: 100%;">
</p>

## Setting up the extension

After [installing](../index.md#installation) the Delphyne extension and opening a workspace containing a Delphyne project (whose root features a [`delphyne.yaml` file](#delphyne-workspace-file)), you can start the Delphyne extension by clicking on the Delphyne icon on the VSCode [activity bar](https://code.visualstudio.com/api/ux-guidelines/activity-bar). Doing so will spawn a Delphyne language server if one is not running already. You can confirm that the language server is running by looking at the `Delphyne` output channel (from the `Output` tab in the panel). See the [Troubleshooting](#troubleshooting) section if you encounter any problem.

!!! info "Locating the language server"
    The Delphyne extension uses the Python distribution currently selected for the workspace by [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) to launch the language server. If no such distribution is configured, you can set it via the `Python: Select Interpreter` command and then restart VSCode.

Once this is done, you can open a demonstration file and start [evaluating demonstrations](#editing-demonstrations).

### Recommended Editor Configuration

For the best experience, we recommend also installing the following VSCode extensions:

- [YAML Language Support](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml)
- [Better Jinja](https://marketplace.visualstudio.com/items?itemName=samuelcolvin.jinjahtml)

In addition, for optimal readability of demonstration files, line wrapping should be activated for YAML files. We recommend doing the same for Jinja files. Finally, we recommend associating the `*.jinja` extension with the `jinja-md` file format. See below for the recommended JSON configuration, which you can copy to your VSCode settings.

<details>
<summary>.vscode/settings.json</summary>
```json
"[yaml]": {
    "editor.wrappingIndent": "indent",
    "editor.wordWrap": "wordWrapColumn",
    "editor.indentSize": 2,
    "editor.wordWrapColumn": 100,
    "editor.rulers": [100],
},
// Wrap template files for readability.
"[jinja-md]": {
    "editor.wrappingIndent": "indent",
    "editor.wordWrap": "wordWrapColumn",
    "editor.wordWrapColumn": 80,
    "editor.indentSize": 2,
    "editor.rulers": [80],
},
// Indicate that the Jinja templates are written in Markdown
"files.associations": {
    "*.jinja": "jinja-md"
}
```
</details>

### The Delphyne Workspace File {#delphyne-workspace-file}

A Delphyne workspace must have a `delphyne.yaml` file as its roots:

```yaml
strategy_dirs: ["."]
modules: ["module_1", "module_2"]
demo_files: ["demo_1", "demo_2"]
```

This files features the following information:

- A list of directories within which strategy files can be found, relative to the root of the workspace.
- A list of module (i.e. file names without extensions) containing those strategies, to be found within those directories.
- A list of demonstration files, to be passed implicitly to all commands (e.g. when running an oracular program).


## Editing Demonstrations {#editing-demonstrations}

## Running Commands

## Navigating Strategy Trees

## Troubleshooting {#troubleshooting}