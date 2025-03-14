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

Once activated, the Delphyne extension recognizes demonstration files via their extension `*.demo.yaml`. A proper YAML schema is automatically loaded and syntax errors are displayed within the editor. To quickly add a new demonstration, you can use the `demo` snippet (by typing `demo` and clicking on ++tab++). All available snippets are listed in [`vscode-ui/snippets.json`](https://github.com/jonathan-laurent/delphyne/blob/main/vscode-ui/snippets.json).

To evaluate a demonstration, put your cursor anywhere in its scope. A light bulb should then appear, suggesting that code actions are available. Use ++cmd+period++ to see available code actions and select `Evaluate Demonstration`. Diagnostics should then appear, possibly after a moment of waiting (in which case a pending task should be displayed in Delphyne's `Tasks` view). If the demonstration was successfully evaluated, an `info` diagnostic should be shown for every test. Otherwise, warnings and errors can be displayed. These diagnostics will stay displayed until the demo file ir closed or the demonstration gets updated. Note that adding comments or modifying other demonstrations does _not_ invalidate them.

Each test in a demonstration, even a failing one, describes a path through the underlying search tree. In order to visualize the endpoint of this path, you can put your cursor on the test and select the `View Test Destination` code action. The resulting node and its context will then be displayed in Delphyne's `Tree`, `Node` and `Actions` view. In the typical case where the test is stuck on a query that is unanswered in the demonstration, one can then click on the `+` icon next to its description (within the `Node` view) to add it to the demonstration (if the query exists already, a `Jump To` icon will be shown instead). The standard workflow is then to add an answer to this query and evaluate the demonstration again.

To evaluate all demonstrations within a file, you can use the `Delphyne: Evaluate All Demonstrations in File` command (use ++cmd+shift+p++ to open the command palette). To see the prompt associated to a query, put your cursor on this query and use the `See Prompt` code action. Doing so will create and run the appropriate [command](#commands) in a new tab.

## Running Commands {#commands}

Many interactions with Delphyne can be performed by executing **commands**. A command can be invoked via a YAML file specifying its name and arguments, starting with header `# delphyne-command`. A standard command is the `run_strategy` command that can be used to run an oracular program by specifying a strategy along with a policy (demonstrations are automatically extracted from the files listed in [delphyne.yaml](#delphyne-workspace-file)). To create a new tab with a template for invoking the `run_strategy` command, you can use `Delphyne: Run Strategy` from the VSCode command palette.

<details>
<summary>A Command Example</summary>
```yaml
# delphyne-command

command: run_strategy
args:
  strategy: prove_program
  args:
    prog: |
      use int.Int

      let main () diverges =
        let ref a = any int in
        let ref b = a * a + 2 in
        let ref x = 0 in
        while b < 50 do
          x <- x * (b - 1) + 1;
          b <- b + 1;
        done;
        assert { x >= 0 }
  policy: prove_program_policy
  policy_args: {}
  num_generated: 1
  budget:
    num_requests: 60
```
</details>

To execute a command, one can put the cursor over it and use the `Execute Command` code action. Doing so will launch a new task that can be viewed in the Delphyne `Task` view. When the task terminates (successfully or with an error), the command's result is appended at the end of the command file (and can be discarded using the `Clear Output` code action). When a command returns a trace, the latter can be [inspected](#navigating-trees) via the `Show Trace` code action.

The progress of commands can be supervised while they are running through the `Tasks` view. This view lists all currently running commands and maps each one to a set of actions. For example, a command can be cancelled or its progress indicator shown on the status bar. In addition, the `Update Source` action (pen icon) allows dumping the command's current _partial_ result to the command file. In the case of the `run_strategy` command, this allows inspecting the current trace _at any point in time_ while search is still undergoing.

!!! note
    In the future, Delphyne will allow adding new commands by registering command scripts. 

## Navigating Strategy Trees {#navigating-trees}

## Tips and Shortcuts

## Troubleshooting {#troubleshooting}