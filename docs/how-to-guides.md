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

There are mainly two ways to run oracular programs:

- **Using [`StrategyInstance.run_toplevel`][delphyne.StrategyInstance.run_toplevel]**: As demonstrated in the [Overview](./manual/overview.md#writing-a-policy), one can manually create a [policy environment][delphyne.PolicyEnv] and extract a [search stream][delphyne.Stream] from a pair of a strategy instance and of a policy. This can be done within any Python script, but by default offers no support for exporting logs and traces, displaying progress, caching LLM requests, handling interruption, etc... Enabling all these features requires substantial additional setup.
- **Running a command**: Instead, a [command file](./manual/extension.md#commands) can be created that specifies a strategy instance, a policy, some search budget along with [extra information][delphyne.stdlib.commands.run_strategy.RunStrategyArgs]. The specified command can be launched from [within VSCode](./manual/extension.md#commands) or from the shell, using the [Delphyne CLI][delphyne.__main__.DelphyneCLI.run] (e.g. `run my_command.exec.yaml --cache --update`).


## Debugging an Oracular Program {#debugging}

Here are some various tips for debugging oracular programs:

- Strategies can be debugged before associated policies are defined by writing demonstrations, for which the Delphyne VSCode extension provides [rich feedback](./manual/extension.md#editing-demonstrations).
- For VSCode to stop at breakpoints in strategies while evaluating demonstrations, the Delphyne server must be started with a debugger attached. See the box below for how to do this. Instructions for killing and starting Delphyne servers is available [here](./manual/extension.md#starting-server). 
- Debugging messages can be included in strategy trees using the [`message`][delphyne.message] function.
- When running oracular programs via [commands](./manual/extension.md#commands), the command output features various useful debugging information by default, in the form of policy logs (including a log of all LLM requests, parsing errors, etc...) and of an [inspectable trace](./manual/extension.md#navigating-trees).
- If request caching is enabled (by specifying the [`cache_file`][delphyne.stdlib.commands.run_strategy.RunStrategyArgs] argument in the command file *or* passing the `--cache` option in the [CLI][delphyne.__main__.DelphyneCLI.run]), then any run of a command can be replayed identically with a debugger attached (see information below), unless a nondeterministic policy is used (e.g. most policies that use multiple threads).

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


## Tuning an Oracular Program {#tuning}

Policies often feature many hyperparameters that must be tuned for the associated oracular program to perform well. The Delphyne standard library defines an [`Experiment`][delphyne.stdlib.experiments.experiment_launcher.Experiment] class for running an oracular program on a set of different hyperparameter combinations. It supports the use of multiple workers, allows interrupting and resuming experiments, retrying failed attempts, and caching all LLM requests for replicability.

For a usage example, see `examples/find_invariants/experiments`.


## Writing a Conversational Agent {#conversational}

A common pattern for interacting with LLMs is to have multi-message exchanges where the full conversation history is resent repeatedly. LLMs are also sometimes allowed to request tool calls. This pattern is implemented by the [`interact`][delphyne.interact] strategy from Delphyne's standard library. For usage examples, see:

- `examples/find_invariants/baseline.py`: real-world example from the [oracular programming paper](https://arxiv.org/abs/2502.05310)
- `tests/example_strategies.py:propose_article`: simple example also involving tool calls

!!! tip "Vertical vs Horizontal LLM Pipelines"
    Delphyne supports the bidirectional integration of two complementary kinds of agents: __vertical__ agents, where a specialized program orchestrates calls to LLMs, and __horizontal__ agents, where an LLM orchestrates calls to tools. Delphyne supports the implementation of horizontal agents via its [`interact`][delphyne.interact] strategy, and allows these agents to invoke tools that are themselves implemented as oracular programs -- whether vertical or horizontal.

## Performing Expensive Computations in Strategies {#compute}

For efficiency and replicability reasons, strategies must not directly perform expensive and possibly nondeterministic computations (e.g. a call to an external SMT solver with a wall clock timeout). In such cases, the [`Compute`][delphyne.Compute] effect should be used. See the [reference page][delphyne.Compute] for details and explanations. For example usage, see `examples/find_invariants/abduct_and_branch.py` and the associated demonstration file.