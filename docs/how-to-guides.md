# How-To Guides

## Creating a New Delphyne Project {#new-project}

To start a new Delphyne project, we recommend taking the following steps:

1. Ensure Delphyne is properly [installed](./index.md#installation).
2. Create a new folder for your project.
3. Add an initially empty [`delphyne.yaml`](./manual/extension.md#config) file in this folder.
4. Ensure that your local or global [VSCode settings](./manual/extension.md#editor-config) are right.
5. Create a Python file and define a strategy inside it.
6. Register this file in the [`modules`](./manual/extension.md#config) section of `delphyne.yaml`.
7. To benefit from typechecking, ensure that Pyright is running in strict mode.
8. Optionally, create a `prompts` folder for storing [Jinja](https://jinja.palletsprojects.com/en/stable/) prompt templates.
9. Create a [demonstration file](./manual/extension.md#editing-demonstrations) with extension `.demo.yaml`.
10. As your project matures, consider adding tests along with a `pyproject.toml` file.

## Running an Oracular Program {#running}

There are mostly two ways to run oracular programs:

- **Using [`StrategyInstance.run_toplevel`][delphyne.StrategyInstance.run_toplevel]**: As demonstrated in the [Overview](./manual/overview.md#writing-a-policy), one can manually create a [policy environment][delphyne.PolicyEnv] and extract a [search stream][delphyne.Stream] from a pair of a strategy instance and of a policy. This can be done within any Python script, but offers by default no support for exporting logs and traces, displaying progress, caching LLM requests, handling interruption, etc... Enabling all these features requires substantial additional setup.
- **Running a command**: Instead, a [command file](./manual/extension.md#commands) can be created that specifies a strategy instance, a policy, some search budget along with [extra information][delphyne.stdlib.commands.run_strategy.RunStrategyArgs]. The specified command can be launched from [within VSCode](./manual/extension.md#commands) or from the shell, using the [Delphyne CLI][delphyne.__main__.DelphyneCLI.run] (e.g. `run my_command.exec.yaml --cache --update`).

## Debugging an Oracular Program {#debugging}

Here are some various tips for debugging oracular programs:

- Strategies can be debugged before associated policies are defined by writing demonstrations, for which the Delphyne VSCode extension provides [rich feedback](./manual/extension.md#editing-demonstrations).
- For VSCode to stop at breakpoints in strategies while evaluating demonstrations, the Delphyne server must be started with a debugger attached. See the box below for how to do this. Instructions for killing and starting Delphyne servers is available [here](./manual/extension.md#starting-server). 
- Debugging messages can be included in strategy trees using the [`message`][delphyne.message] function.
- When running oracular programs via [commands](./manual/extension.md#commands), the command output features various useful debugging information by default, in the form of policy logs (including a log of all LLM requests, parsing errors, etc...) and of an [inspectable trace](./manual/extension.md#navigating-trees).
- If request caching is enabled (by specifying the [`cache_dir`][delphyne.stdlib.commands.run_strategy.RunStrategyArgs] argument in the command file *or* passing the `--cache` option in the [CLI][delphyne.__main__.DelphyneCLI.run]), then any run of a command can be replayed identically with a debugger attached (see information below), unless a nondeterministic policy is used (e.g. most policies that use multiple threads).

!!! tip "Attaching a Debugger to the Delphyne CLI"

    For debugging purposes, it is useful to attach a Python debugger to the [Delphyne CLI][delphyne.__main__.DelphyneCLI]. To do so, we recommend defining the following `debug-delphyne` alias:
    
    ```sh
    alias debug-delphyne='python -m debugpy --listen 5678 --wait-for-client -m delphyne'
    ```

    In addition, you should add the following to your `.vscode/launch.json` file.

    ```json
    "name": "Attach",
    "type": "debugpy",
    "request": "attach",
    "connect": {
        "host": "localhost",
        "port": 5678
        }
    }
    ```
    You can then run the Delphyne CLI with a debugger attached by simply substituting `delphyne` with `debug-delphyne`. For example, `debug-delphyne serve` launches a Delphyne server with a debugger attached. Note that the server does not immediately start after running this command, which waits for the user to launch the `Attach` debugging profile from inside VSCode.

<!-- To debug the language server or even specific strategies, it is useful to attach a debugger to the language server. To do so, you should open VSCode at the root of the Delphyne repository and use the `Debug Server` debugging profile. This will start the server in debug mode (on port 8000). Starting the Delphyne extension when a server instance is running already will cause the extension to use this instance (as confirmed by the log output in the `Delphyne` channel). You can then put arbitrary breakpoints in the server source code or even in strategy code. -->

## Tuning an Oracular Program {#tuning}

## Writing a Conversational Agent {#conversational}

## Performing Expensive Computations in Strategies {#compute}