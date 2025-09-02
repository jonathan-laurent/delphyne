# The Delphyne VSCode Extension

A _Visual Studio Code_ extension is available for interactively writing demonstrations, navigating strategy trees, and running oracular programs.

<p align="center">
  <img src="../../assets/screenshot/extension-screenshot/dark.png#only-dark" alt="Extension Screenshot" style="width: 100%;">
  <img src="../../assets/screenshot/extension-screenshot/light.png#only-light" alt="Extension Screenshot" style="width: 100%;">
</p>

## Setting Up The Extension

### Recommended Editor Configuration {#editor-config}

We provide instructions for installing the Delphyne extension on the documentation [home page](../index.md#installation). For the best experience, we recommend also installing the following VSCode extensions:

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

### Starting The Delphyne Server {#starting-server}

The Delphyne extension relies on Delphyne's _language server_ to [evaluate demonstrations](#editing-demonstrations) and [run commands](#commands). The language server is automatically started in the background when the Delphyne extension is activated (by clicking on the Delphyne icon on the VSCode [activity bar](https://code.visualstudio.com/api/ux-guidelines/activity-bar)), if it is not running already (listening on port 3008).

!!! info "Locating the language server"
    The Delphyne extension uses the Python distribution currently selected for the workspace by [Pylance](https://marketplace.visualstudio.com/items?itemName=ms-python.vscode-pylance) to launch the language server. If no such distribution is configured, you can set it via the `Python: Select Interpreter` command and then try again using the `Delphyne: Start Server Command`.

The background server process can be shut down using the `Delphyne: Kill Server` command. The `Delphyne: Start Server Command` can be used to restart it, if it is not running already. The server can be launched outside VSCode using the [`delphyne serve`][delphyne.__main__.DelphyneCLI.serve] shell command. Since the server is stateless, it can be restarted at any time. One reason for running the server outside VSCode is to [attach a debugger to it](../how-to-guides.md#debugging), which allows setting breakpoints inside strategies, policies, or even inside the demonstration interpreter.

 You can confirm that the language server is running by looking at the `Delphyne` output channel (from the `Output` tab in the panel). See [Troubleshooting](#troubleshooting) if you encounter any problem.

### Detecting Project Root Directories {#project-root}

When [evaluating a demonstration](#editing-demonstrations) or [running a command](#commands), the Delphyne extension locates the corresponding project root directory as follows:

1. If a transitive parent directory for the current demonstration or command file contains a `delphyne.yaml` configuration file, it is selected.
2. If the current editor is not attached to an existing file[^anon-file] or no `delphyne.yaml` file can be found, the current VSCode's workspace directory is used.

The `Delphyne: Show Root Directory for Current File` command can be used to show the current project root. The rules above allow browsing and editing multiple Delphyne projects within a single VSCode workspace.

[^anon-file]: For example, it may contain an unsaved command specification.

### Global and Local Configuration {#config}

Both [demonstration](#editing-demonstrations) and [command](#commands) files are evaluated in the context of a given [configuration record][delphyne.CommandExecutionContext], which specifies information such as the location and names of Python modules in which strategies can be found, the location of prompting templates and demonstration files, etc... This information can be stored in the project's `delphyne.yaml` file, whose content may look like:

```yaml
strategy_dirs: ["."]
modules: ["module_1", "module_2"]
demo_files: ["demo_1", "demo_2"]
```

See the [Reference][delphyne.CommandExecutionContext] for the list and description of all available settings. All settings have default values so empty `delphyne.yaml` files are allowed (or no file at all if the project root coincides with the VSCode workspace). In addition, any subset of global settings from the `delphyne.yaml` file can be locally overriden in individual demonstration or command files by prefixing it with a [`@config` comment block][delphyne.CommandExecutionContext].

## Editing Demonstrations {#editing-demonstrations}

Once activated, the Delphyne extension recognizes demonstration files via their extension `*.demo.yaml`. A proper YAML schema is automatically loaded and syntax errors are displayed within the editor. To quickly add a new demonstration, you can use the `demo` snippet (by typing `demo` and clicking on ++tab++). All available snippets are listed in [`vscode-ui/snippets.json`](https://github.com/jonathan-laurent/delphyne/blob/main/vscode-ui/snippets.json).

To evaluate a demonstration, put your cursor anywhere in its scope. A light bulb should then appear, suggesting that code actions are available. Use ++cmd+period++ to see available code actions and select `Evaluate Demonstration`. Diagnostics should then appear, possibly after a moment of waiting (in which case a pending task should be displayed in Delphyne's `Tasks` view). If the demonstration was successfully evaluated, an `info` diagnostic should be shown for every test. Otherwise, warnings and errors can be displayed. These diagnostics will stay displayed until the demo file is closed or the demonstration gets updated. Note that adding comments or modifying other demonstrations does _not_ invalidate them.

Each test in a demonstration, even a failing one, describes a path through the underlying search tree. In order to visualize the endpoint of this path, you can put your cursor on the test and select the `View Test Destination` code action. The resulting node and its context will then be displayed in Delphyne's `Tree`, `Node` and `Actions` view. In the typical case where the test is stuck on a query that is unanswered in the demonstration, one can then click on the `+` icon next to its description (within the `Node` view) to add it to the demonstration (if the query exists already, a `Jump To` icon will be shown instead). The standard workflow is then to add an answer to this query and evaluate the demonstration again.

To evaluate all demonstrations within a file, you can use the `Delphyne: Evaluate All Demonstrations in File` command (use ++cmd+shift+p++ to open the command palette). To see the prompt associated to a query, put your cursor on this query and use the `See Prompt` code action. Doing so will create and run the appropriate [command](#commands) in a new tab.

!!! info "Automatic Reloading of Strategies"
    The language server reloads all modules listed in [`delphyne.yaml`](#config) for _every_ query, using `importlib.reload`. This way, strategies can be updated interactively without effort. Note that modules are reloaded in the order in which they are listed. Thus, a module should always be listed after its dependencies.

!!! info "Evaluating Demonstrations using the CLI"
    Demonstrations can also be evaluated from the shell, using the [Delphyne CLI][delphyne.__main__.DelphyneCLI]. However, the CLI provides much more limited feedback so it is mainly useful for testing and continuous integration.

## Running Commands {#commands}

Many interactions with Delphyne can be performed by executing **commands**.Commands can be specified in YAML files with extension `.exec.yaml`. Unnamed documents are also recognized as command files if they start with line `# delphyne-command`, possibly following other YAML comments. Commands can emit diagnostics and intermediate status updates, output a stream of partial results and be safely interrupted. The Delphyne extension provides editor support for these features, via its `Task View`. Commands can also be run from the [Delphyne CLI][delphyne.__main__.DelphyneCLI], which is useful for specifying test suites or launching a large number of commands. 

A standard command is [`run_strategy`][delphyne.stdlib.commands.run_strategy.run_strategy], which can be used to run an oracular program by specifying a strategy along with a policy (demonstrations are automatically extracted from the files listed in [delphyne.yaml](#config)). To create a new tab with a template for invoking the `run_strategy` command, you can use `Delphyne: Run Strategy` from the VSCode command palette.

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

To execute a command, one can put the cursor over it and use the `Execute Command` code action. Doing so will launch a new task that can be viewed in the Delphyne `Task` view. When the task terminates (successfully or with an error), the command's result is appended at the end of the command file (and can be discarded using the `Clear Output` code action). When a command returns a [trace][delphyne.Trace], the latter can be [inspected](#navigating-trees) via the `Show Trace` code action.

The progress of commands can be supervised while they are running through the `Tasks` view. This view lists all currently running commands and maps each one to a set of actions. For example, a command can be cancelled or its progress indicator shown on the status bar. In addition, the `Update Source` action (pen icon) allows dumping the command's current _partial_ result to the command file. In the case of the `run_strategy` command, this allows inspecting the current [trace][delphyne.Trace] (i.e. the set of all visited tree nodes and spaces) _at any point in time_ while search is still undergoing.

!!! note
    In the future, Delphyne will allow adding new commands by registering command scripts. 


## Navigating Strategy Trees {#navigating-trees}

Whether they originate from evaluating demonstrations or running commands, [traces][delphyne.Trace] can be inspected using the `Tree`, `Node` and `Actions` views (see screenshot at the top of this page). These views are synchronized together and display information about a single node at a time. The `Tree` view indicates a path from the root to the current node and allows jumping to every intermediate node on this path. The `Node` view shows the node type and all associated spaces. For each space, it shows the underlying query or allows jumping to the underlying tree. Finally, the `Actions` view lists all children of the current node that belong to the trace. Actions leading to subtrees containing success nodes are indicated by small checkmarks.

Navigation operations can be undone by clicking on the `Undo` icon on the header of the tree view or by using shortcut ++cmd+d++ followed by ++cmd+z++. 


## Tips and Shortcuts

- Folding is very useful to keep demonstration readable. You should learn the standard [VSCode shortcuts](https://code.visualstudio.com/docs/editor/codebasics#_folding) for controlling folding (++cmd+k+cmd+l++ for toggling folding under the cursor, ++cmd+k+cmd+0++ for folding everything, ++cmd+k+cmd+3++ for folding everything at depth 3, ++cmd+k+cmd+j++ for unfolding everything...). In addition, the custom Delphyne shortcut ++cmd+d+cmd+k++ can be used to fold all strategy and query arguments outside of the cursor's scope.
- Shortcut ++cmd+d+cmd+v++ can be used to focus on Delphyne's views.
- The Github Copilot Code Actions can get in the way of using Delphyne and can be disabled with the `"github.copilot.editor.enableCodeActions": false` setting.
- When editing YAML files (and demonstration files in particular), VSCode allows semantically expanding and shrinking the selection with respect to the underlying syntax tree via ++cmd+ctrl+shift+arrow-left++ and ++cmd+ctrl+shift+arrow-right++.
- To close a tab (and in particular a command tab), you can use shortcut ++cmd+w++, followed by ++cmd+d++ to decline saving the tab's content if prompted.
- Since command outputs can be verbose, it is recommended to start folding sections starting at level 4 (using ++cmd+k+cmd+4++) and then unfolding sections as needed.

## Troubleshooting {#troubleshooting}

### Accessing log information

The Delphyne extension outputs logging information in three different output channels:

- `Delphyne`: main logging channel
- `Delphyne Server`: the output of the language server is redirected here (only when the server was started by the extension)
- `Delphyne Tasks`: displays the logging information produced by commands such as `run_strategy`

You should consult those channels if anything goes wrong. Also, to output more information, you can ask the extension to log more information by raising the log level to `Debug` or `Trace` via the `Developer: Set Log Level` command.

### Killing a server instance still running in the background

After VSCode quit unexpectedly, the language server may still be running in background, which may cause problems when trying to restart the extension. On Unix systems, the language server can be killed by killing the program listening to port 3008:

```sh
sudo kill -9 $(sudo lsof -t -i :3008)
```