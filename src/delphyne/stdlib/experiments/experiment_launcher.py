"""
A utility class for defining, launching and managing experiments.
"""

import json
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol, Self

import fire  # type: ignore
import pandas as pd  # type: ignore
import yaml

import delphyne.stdlib.commands as cmd
import delphyne.stdlib.models as md
from delphyne.stdlib.tasks import CommandExecutionContext, run_command
from delphyne.utils.typing import NoTypeInfo, pydantic_dump, pydantic_load

type _ModelWrapper = Callable[[md.LLM], md.LLM]


EXPERIMENT_STATE_FILE = "experiment.yaml"
STATUS_FILE = "statuses.txt"
RESULT_FILE = "result.yaml"
LOG_FILE = "log.txt"
EXCEPTION_FILE = "exception.txt"
CACHE_FILE = "cache.yaml"
RESULTS_SUMMARY = "results_summary.csv"


@dataclass
class ConfigInfo[Config]:
    """
    Information stored in the persistent configuration state for each
    configuration.

    Attributes:
        params: The configuration.
        status: Status of the configuration.
    """

    params: Config
    status: Literal["todo", "done", "failed"]


@dataclass
class ExperimentState[Config]:
    """
    Persistent state of an experiment, stored on disk as a YAML file.
    """

    name: str | None
    description: str | None
    configs: dict[str, ConfigInfo[Config]]

    def inverse_mapping(self) -> Callable[[Config], str | None]:
        """
        Compute an inverse function mapping configurations to their
        unique names (or None if not in the state).
        """
        tab: dict[str, str] = {}
        for name, info in self.configs.items():
            tab[_config_unique_repr(info.params)] = name

        def reverse(config: Config) -> str | None:
            return tab.get(_config_unique_repr(config), None)

        return reverse


class ExperimentFun[Config](Protocol):
    """
    A function defining an experiment, which maps a configuration (i.e.,
    a set of parameters) to a set of arguments for the `run_strategy`
    command. Note that caching-related arguments do not need to be set
    since they are overriden by the `Experiment` class.
    """

    def __call__(self, config: Config, /) -> cmd.RunStrategyArgs: ...


@dataclass(kw_only=True)
class Experiment[Config]:
    """
    An experiment that consists in running an oracular program on a set of
    different hyperparameter combinations.

    This class allows defining and running experiments. It supports the
    use of multiple workers, and allows interrupting and resuming
    experiments (the persistent experiment state is stored in a file on
    disk). Failed configurations can be selectively retried. By
    activating caching, a successful experiment can be replicated (or
    some of its configurations replayed with a debugger) without issuing
    calls to LLMs or to tools with non-replicable outputs.

    Type Parameters:
        Config: Type parameter for the configuration type, which is a
            dataclass that holds all experiment hyperparameters.

    Attributes:
        experiment: The experiment function, which defines a run of an
            oracular program for each configuration.
        output_dir: The directory where all experiment data is stored
            (persistent state, results, logs, caches...). The directory
            is created if it does not alredy exist.
        context: Command execution context, which contains the kind of
            information usually provided in the `delphyne.yaml` file
            (experiments do not recognize such files). Note that the
            `cache_root` argument should not be set, since it is
            disregarded and overriden by the `Experiment` class.
        configs: A sequence of configurations to run. If `None` is
            provided and the experiment already has a persistent state
            stored on disk, the list of configurations is loaded from
            there upon loading.
        config_type: The `Config` type, which is either passed
            explicitly or deduced from the `configs` argument.
        name: Experiment name, which is stored in the persistent state
            file when provided and is otherwise not used.
        description: Experiment description, which is stored in the
            persistent state file when provided and is otherwise not used.
        config_naming: A function for attributing string identifiers to
            configurations, which maps a configuration along with a
            fresh UUID to a name. By default, the UUID alone is used.
        cache_requests: Whether or not to enable caching of LLM requests
            and expensive computations (see `Compute`). When this is
            done, the experiment can be reliably replicated, without
            issuing LLM calls.
        export_raw_trace: Whether to export the raw trace for all
            configuration runs.
        export_log: Whether to export the log messages for all
            configuration runs.
        export_browsable_trace: Whether to export a browsable trace for
            all configuration runs, which can be visualized in the VSCode
            extension (see `delphyne.analysis.feedback.Trace`). However,
            such traces can be large.

    ## Tips

    - New hyperparameters can be added to the `Config` type without
      invalidating an existing experiment's persistent state, by
      providing default values for them.
    """

    experiment: ExperimentFun[Config]
    output_dir: Path  # absolute path expected
    context: CommandExecutionContext
    configs: Sequence[Config] | None = None
    config_type: type[Config] | NoTypeInfo = NoTypeInfo()
    name: str | None = None
    description: str | None = None
    config_naming: Callable[[Config, uuid.UUID], str] | None = None
    cache_requests: bool = True
    export_raw_trace: bool = True
    export_log: bool = True
    export_browsable_trace: bool = True

    def __post_init__(self):
        # We override the cache root directory.
        assert self.context.cache_root is None
        self.context = replace(self.context, cache_root=self.output_dir)
        if isinstance(self.config_type, NoTypeInfo):
            if self.configs:
                self.config_type = type(self.configs[0])

    def load(self) -> Self:
        """
        Load the experiment.

        If no persistent state exists on disk, it is created (with all
        configurations marked with "todo" status). If some experiment
        state exists on disk, it is loaded. If more configurations are
        specified in `self.configs` than are specified on disk, the
        missing configurations are added to the persistent state and
        marked with "todo". If the persistent state contains
        configurations that are not specified in `self.configs`, a
        warning is shown. Use the `clean_index` method to remove these
        configurations from the persistent state.

        Return `self`, so as to allow chaining.
        """
        if not self._dir_exists():
            # If we create the experiment for the first time
            print(f"Creating experiment directory: {self.output_dir}.")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            state = ExperimentState[Config](self.name, self.description, {})
            self._save_state(state)
        if self.configs is not None:
            self._add_configs_if_needed(self.configs)
            # Print a warning if the state on disk features additional configs.
            state = self._load_state()
            assert state is not None
            assert len(self.configs) <= len(state.configs)
            if len(self.configs) < len(state.configs):
                print(
                    f"Warning: {len(state.configs) - len(self.configs)} "
                    "additional configuration(s) found in the state."
                )
        return self

    def is_done(self) -> bool:
        """
        Check if the experiment is done, i.e., all configurations are
        marked as "done".
        """
        state = self._load_state()
        assert state is not None
        return all(info.status == "done" for info in state.configs.values())

    def clean_index(self) -> None:
        """
        Remove from the persistent state file all configurations that
        are not mentioned in `self.configs`.
        """
        state = self._load_state()
        assert state is not None
        assert self.configs is not None
        in_config = set(_config_unique_repr(c) for c in self.configs)
        to_delete = [
            c
            for c, i in state.configs.items()
            if _config_unique_repr(i.params) not in in_config
        ]
        print(f"Removing {len(to_delete)} configuration(s) from the state.")
        for c in to_delete:
            del state.configs[c]
        self._save_state(state)

    def mark_errors_as_todos(self):
        """
        Update the persistent state to mark all configurations with
        status "failed" as "todo". They will be retried when the
        `resume` method is called.
        """
        state = self._load_state()
        assert state is not None
        for _, info in state.configs.items():
            if info.status == "failed":
                info.status = "todo"
        self._save_state(state)

    def resume(self, max_workers: int = 1, log_progress: bool = True):
        """
        Resume the experiment, running all configurations with state
        "todo". Every configuration run results in marking the
        configuration's state with either "failed" (in case an uncaught
        exception was raised) or "done".

        The whole process can be interrupted using Ctrl-C, in which case
        the persistent experiment state is stored on disk, a message is
        printed saying so, and Ctrl-C can be hit again until all workers
        are successfully terminated.

        A summary file is produced at the end of the experiment using
        the `summary_file` method if all configurations were run
        successfully.

        Attributes:
            max_workers: Number of parallel process workers to use.
            log_progress: Whether to show a progress bar in the console.
        """
        state = self._load_state()
        assert state is not None
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_config,
                    context=self.context,
                    experiment=self.experiment,
                    config_name=name,
                    config_dir=self._config_dir(name),
                    config=info.params,
                    cache_requests=self.cache_requests,
                    export_raw_trace=self.export_raw_trace,
                    export_log=self.export_log,
                    export_browsable_trace=self.export_browsable_trace,
                )
                for name, info in state.configs.items()
                if info.status == "todo"
            ]
            if log_progress:
                _print_progress(state)
            try:
                for future in as_completed(futures):
                    name, success = future.result()
                    state.configs[name].status = (
                        "done" if success else "failed"
                    )
                    if log_progress:
                        _print_progress(state)
                self._save_state(state)
                all_successes = all(
                    info.status == "done" for info in state.configs.values()
                )
                if all_successes:
                    print(
                        "\nExperiment successful.\nProducing summary file..."
                    )
                    self.save_summary()
                else:
                    print("\nWarning: some configurations failed.")
            except KeyboardInterrupt:
                print("\nExperiment interrupted. Saving state...")
                self._save_state(state)
                print("State saved.")

    def replay_config_by_name(self, config_name: str) -> None:
        """
        Replay a configuration with a given name, reusing the cache if
        it exists.

        This way, one can debug the execution of an experiment after the
        fact, without any LLMs being called.
        """
        state = self._load_state()
        assert state is not None
        assert config_name is not None
        info = state.configs[config_name]
        assert info.status == "done"
        cmdargs = self.experiment(info.params)
        cmdargs.cache_file = config_name + "/" + CACHE_FILE
        cmdargs.cache_mode = "replay"
        run_command(
            command=cmd.run_strategy,
            args=cmdargs,
            ctx=self.context,
            dump_statuses=None,
            dump_result=None,
            dump_log=None,
        )

    def replay_config(self, config: Config) -> None:
        """
        Replay a configuration. See `replay_config_by_name` for details.
        """
        config_name = self._existing_config_name(config)
        assert config_name is not None
        self.replay_config_by_name(config_name)

    def replay_all_configs(self):
        """
        Replay all configurations, replicating the experiment.
        """
        state = self._load_state()
        assert state is not None
        for config_name in state.configs:
            print(f"Replaying configuration: {config_name}...")
            self.replay_config_by_name(config_name)

    def save_summary(self, ignore_missing: bool = False):
        """
        Save a summary of the results in a CSV file.

        Arguments:
            ignore_missing: If `True`, configurations whose status is
                "failed" or "todo" are ignored. Otherwise, an error is
                raised.
        """

        data = _results_summary(self.output_dir, ignore_missing)
        frame = pd.DataFrame(data)
        summary_file = self.output_dir / RESULTS_SUMMARY
        frame.to_csv(summary_file, index=False)  # type: ignore

    def load_summary(self):
        """
        Load the summary file into a DataFrame.

        The summary file should have been created before using the
        `save_summary` method.
        """

        summary_file = self.output_dir / RESULTS_SUMMARY
        data = pd.DataFrame, pd.read_csv(summary_file)  # type: ignore
        return data

    def get_status(self) -> dict[str, int]:
        """
        Get the status of the experiment configurations.

        Returns:
            A dictionary with keys 'todo', 'done', 'failed' and their
            counts (i.e., number of configurations with this status).
        """
        state = self._load_state()
        assert state is not None
        statuses = state.configs.values()
        num_todo = sum(1 for c in statuses if c.status == "todo")
        num_done = sum(1 for c in statuses if c.status == "done")
        num_failed = sum(1 for c in statuses if c.status == "failed")
        return {"todo": num_todo, "done": num_done, "failed": num_failed}

    def run_cli(self):
        """
        Run a CLI application that allows controlling the experiment
        from the shell. See `ExperimentCLI` for details.
        """
        fire.Fire(ExperimentCLI(self))  # type: ignore

    def _config_dir(self, config_name: str) -> Path:
        return self.output_dir / config_name

    def _add_configs_if_needed(self, configs: Sequence[Config]) -> None:
        state = self._load_state()
        assert state is not None
        rev = state.inverse_mapping()
        num_added = 0
        for c in configs:
            existing_name = rev(c)
            if existing_name is not None:
                continue
            pass
            num_added += 1
            id = uuid.uuid4()
            if self.config_naming is not None:
                name = self.config_naming(c, id)
            else:
                name = str(id)
            state.configs[name] = ConfigInfo(c, status="todo")
        if num_added > 0:
            print(f"Adding {num_added} new configuration(s).")
        self._save_state(state)

    def _dir_exists(self) -> bool:
        return self.output_dir.exists() and self.output_dir.is_dir()

    def _state_type(self) -> type[ExperimentState[Config]]:
        assert not isinstance(self.config_type, NoTypeInfo), (
            "Please set `Experiment.config_type`."
        )
        return ExperimentState[self.config_type]

    def _load_state(self) -> ExperimentState[Config] | None:
        with open(self.output_dir / EXPERIMENT_STATE_FILE, "r") as f:
            parsed = yaml.safe_load(f)
            return pydantic_load(self._state_type(), parsed)

    def _save_state(self, state: ExperimentState[Config]) -> None:
        with open(self.output_dir / EXPERIMENT_STATE_FILE, "w") as f:
            to_save = pydantic_dump(self._state_type(), state)
            yaml.safe_dump(to_save, f, sort_keys=False)

    def _existing_config_name(self, config: Config) -> str | None:
        state = self._load_state()
        assert state is not None
        for name, info in state.configs.items():
            if info.params == config:
                return name
        return None


EXPORTED_BUDGET_FIELDS = [
    "num_completions",
    "num_requests",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "price",
]


def _results_summary(
    exp_dir: Path, ignore_missing: bool = False
) -> Sequence[dict[str, Any]]:
    # Load the experiment state as a python dict
    state_file = exp_dir / EXPERIMENT_STATE_FILE
    with open(state_file, "r") as f:
        parsed = yaml.safe_load(f)
    res: list[dict[str, Any]] = []
    for name, info in parsed["configs"].items():
        params = info["params"]
        if info["status"] != "done":
            assert ignore_missing, f"Missing result for {name}."
            continue
        # Open the result file and parse it in yaml
        result_file = exp_dir / name / RESULT_FILE
        with open(result_file, "r") as f:
            # Read the prefix of the file until a line is encountered
            # starting with `raw_trace` preceded by four spaces. Indeed,
            # the whole YAML file might be huge!
            prefix = ""
            while line := f.readline():
                if line.startswith("    raw_trace"):
                    break
                prefix += line
            else:
                assert False
            result = yaml.safe_load(prefix)
        result = result["outcome"]["result"]
        success = result["success"]
        assert isinstance(success, bool)
        spent = result["spent_budget"]
        price = spent.get("price", 0.0)
        assert isinstance(price, float)
        entry = params | {"success": success}
        for field in EXPORTED_BUDGET_FIELDS:
            entry[field] = spent.get(field, 0)
        res.append(entry)
    return res


def _print_progress(state: ExperimentState[Any]) -> None:
    num_done = sum(1 for c in state.configs.values() if c.status != "todo")
    num_failed = sum(1 for c in state.configs.values() if c.status == "failed")
    num_total = len(state.configs)
    msg = f"\rDone: {num_done} / {num_total}, Failed: {num_failed}"
    print(msg + 40 * " ", end="")


def _run_config[Config](
    context: CommandExecutionContext,
    experiment: ExperimentFun[Config],
    config_name: str,
    config_dir: Path,
    config: Config,
    cache_requests: bool,
    export_raw_trace: bool,
    export_log: bool,
    export_browsable_trace: bool,
) -> tuple[str, bool]:
    cache_file = None
    if cache_requests:
        cache_file = config_dir / CACHE_FILE
        if cache_file.exists():
            cache_file.unlink(missing_ok=True)
    for f in (STATUS_FILE, RESULT_FILE, LOG_FILE):
        file_path = config_dir / f
        if file_path.exists():
            file_path.unlink(missing_ok=True)
    cmdargs = experiment(config)
    if cache_requests:
        # A relative path is expected!
        cmdargs.cache_file = config_name + "/" + CACHE_FILE
    cmdargs.cache_mode = "create"
    cmdargs.export_browsable_trace = export_browsable_trace
    cmdargs.export_log = export_log
    cmdargs.export_raw_trace = export_raw_trace
    try:
        run_command(
            command=cmd.run_strategy,
            args=cmdargs,
            ctx=context,
            dump_statuses=config_dir / STATUS_FILE,
            dump_result=config_dir / RESULT_FILE,
            dump_log=config_dir / LOG_FILE,
            add_header=True,
        )
        success = True
    except Exception:
        config_dir.mkdir(parents=True, exist_ok=True)
        with open(config_dir / EXCEPTION_FILE, "w") as f:
            import traceback

            traceback.print_exc(file=f)
        success = False
    return (config_name, success)


def _config_unique_repr(config: object):
    # We want a unique representation for the configuration We start
    # doing a round-trip to ensure that Config(1) and Config(1.0) are
    # treated as equal.
    cls = type(config)
    python = pydantic_dump(cls, config)
    config = pydantic_load(cls, python)
    python = pydantic_dump(cls, config)
    return json.dumps(python, sort_keys=True)


class ExperimentCLI:
    """
    A CLI application for controlling an experiment from the shell.
    """

    def __init__(self, experiment: Experiment[Any]):
        self.experiment = experiment

    def __call__(self):
        """
        If no argument is provided, the `run` method is called.
        """
        self.run()

    def run(
        self,
        *,
        max_workers: int = 1,
        retry_errors: bool = False,
        cache: bool = True,
        verbose_output: bool = False,
    ):
        """
        Start or resume the experiment.

        Attributes:
            max_workers: Number of parallel process workers to use.
            retry_errors: Mark failed configurations to be retried.
            cache: Enable caching of LLM requests and potentially
                non-replicable computations.
            verbose_output: Export raw traces and browsable traces in
                result files, enabling inspection by the Delphyne VSCode
                extension's tree view.
        """
        self.experiment.cache_requests = cache
        self.experiment.export_raw_trace = verbose_output
        self.experiment.export_browsable_trace = verbose_output
        self.experiment.export_log = True

        self.experiment.load()
        if retry_errors:
            self.experiment.mark_errors_as_todos()
        self.experiment.resume(max_workers=max_workers)

    def status(self):
        """
        Print the status of the experiment.
        """
        status_counts = self.experiment.get_status()
        print(
            f"Experiment '{self.experiment.name}':\n"
            f"  - {status_counts['todo']} configurations to do\n"
            f"  - {status_counts['done']} configurations done\n"
            f"  - {status_counts['failed']} configurations failed"
        )

    def replay(self, config: str | None = None):
        """
        Replay one or all configurations.

        Arguments:
            config: The name of the configuration to replay. If not
                provided, all configurations are replayed.
        """
        self.experiment.load()
        if config is None:
            self.experiment.replay_all_configs()
        else:
            self.experiment.replay_config_by_name(config)

    def clean_index(self):
        """
        Clean unregistered configurations from the persistent state
        file.
        """
        self.experiment.load().clean_index()

    def force_summary(self):
        """
        Force the generation of a summary file, even if not all
        configurations were successfully run.
        """
        self.experiment.load().save_summary(ignore_missing=True)
