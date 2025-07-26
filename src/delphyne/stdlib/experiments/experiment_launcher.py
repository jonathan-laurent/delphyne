"""
Utilities to launch experiments in Delphyne.


An experiment is associated a directory.
"""

import json
import shutil
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import fire  # type: ignore
import pandas as pd  # type: ignore
import yaml

import delphyne.analysis as analysis
import delphyne.stdlib.commands as cmd
import delphyne.stdlib.models as md
from delphyne.stdlib.tasks import CommandExecutionContext, run_command
from delphyne.utils.typing import pydantic_dump, pydantic_load

type _ModelWrapper = Callable[[md.LLM], md.LLM]


EXPERIMENT_STATE_FILE = "experiment.yaml"
STATUS_FILE = "statuses.txt"
RESULT_FILE = "result.yaml"
LOG_FILE = "log.txt"
EXCEPTION_FILE = "exception.txt"
CACHE_DIR = "llm_cache"
RESULTS_SUMMARY = "results_summary.csv"


@dataclass
class ConfigInfo[Config]:
    params: Config
    status: Literal["todo", "done", "failed"]


@dataclass
class ExperimentState[Config]:
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


class _ExperimentFun[Config](Protocol):
    """
    The `requests_cache` argument is overriden.
    """

    def __call__(self, config: Config) -> cmd.RunStrategyArgs: ...


@dataclass
class Experiment[Config]:
    """
    The `context.requests_cache_dir` argument is overriden.
    """

    dir: Path
    context: CommandExecutionContext
    experiment: _ExperimentFun[Config]
    config_type: type[Config]
    configs: Sequence[Config] | None = None
    name: str | None = None
    description: str | None = None
    config_naming: Callable[[Config, uuid.UUID], str] | None = None
    cache_requests: bool = True
    export_raw_trace: bool = True
    export_log: bool = True
    export_browsable_trace: bool = True

    def __post_init__(self):
        # We override the cache requests directory.
        assert self.context.requests_cache_dir is None
        self.context = replace(self.context, requests_cache_dir=self.dir)

    def load(self):
        if not self.dir_exists():
            # If we create the experiment for the first time
            print(f"Creating experiment directory: {self.dir}.")
            self.dir.mkdir(parents=True, exist_ok=True)
            state = ExperimentState[Config](self.name, self.description, {})
            self.save_state(state)
        if self.configs is not None:
            self.add_configs_if_needed(self.configs)
            # Print a warning if the state on disk features additional configs.
            state = self.load_state()
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
        state = self.load_state()
        assert state is not None
        return all(info.status == "done" for info in state.configs.values())

    def config_dir(self, config_name: str) -> Path:
        return self.dir / config_name

    def clean_index(self) -> None:
        """
        Remove from the state file all configurations that are not
        explicitly mentioned in `self.configs`.
        """
        state = self.load_state()
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
        self.save_state(state)

    def add_configs_if_needed(self, configs: Sequence[Config]) -> None:
        state = self.load_state()
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
        self.save_state(state)

    def dir_exists(self) -> bool:
        return self.dir.exists() and self.dir.is_dir()

    def state_type(self) -> type[ExperimentState[Config]]:
        return ExperimentState[self.config_type]

    def load_state(self) -> ExperimentState[Config] | None:
        with open(self.dir / EXPERIMENT_STATE_FILE, "r") as f:
            parsed = yaml.safe_load(f)
            return pydantic_load(self.state_type(), parsed)

    def save_state(self, state: ExperimentState[Config]) -> None:
        with open(self.dir / EXPERIMENT_STATE_FILE, "w") as f:
            to_save = pydantic_dump(self.state_type(), state)
            yaml.safe_dump(to_save, f, sort_keys=False)

    def resume(self, max_workers: int = 1, log_progress: bool = True):
        state = self.load_state()
        assert state is not None
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_config,
                    context=self.context,
                    experiment=self.experiment,
                    config_name=name,
                    config_dir=self.config_dir(name),
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
                self.save_state(state)
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
                self.save_state(state)
                print("State saved.")

    def mark_errors_as_todos(self):
        state = self.load_state()
        assert state is not None
        for _, info in state.configs.items():
            if info.status == "failed":
                info.status = "todo"
        self.save_state(state)

    def existing_config_name(self, config: Config) -> str | None:
        state = self.load_state()
        assert state is not None
        for name, info in state.configs.items():
            if info.params == config:
                return name
        return None

    def replay_config_by_name(self, config_name: str) -> None:
        """
        Replay a configuration, reusing the cache if it exists.

        This way, one can debug the execution of an experiment after the
        fact, without any LLMs being called.
        """
        state = self.load_state()
        assert state is not None
        assert config_name is not None
        info = state.configs[config_name]
        assert info.status == "done"
        cmdargs = self.experiment(info.params)
        cmdargs.requests_cache = CACHE_DIR
        cmdargs.requests_cache_mode = "replay"
        run_command(
            command=cmd.run_strategy,
            args=cmdargs,
            ctx=self.context,
            dump_statuses=None,
            dump_result=None,
            dump_log=None,
        )

    def replay_config(self, config: Config) -> None:
        config_name = self.existing_config_name(config)
        assert config_name is not None
        self.replay_config_by_name(config_name)

    def replay_all_configs(self):
        state = self.load_state()
        assert state is not None
        for config_name in state.configs:
            print(f"Replaying configuration: {config_name}...")
            self.replay_config_by_name(config_name)

    def save_summary(self, ignore_missing: bool = False):
        """
        Save a summary of the results in a CSV file.
        """

        data = results_summary(self.dir, ignore_missing)
        frame = pd.DataFrame(data)
        summary_file = self.dir / RESULTS_SUMMARY
        frame.to_csv(summary_file, index=False)  # type: ignore

    def load_summary(self):
        """
        Load the summary of the results in a DataFrame.
        """

        summary_file = self.dir / RESULTS_SUMMARY
        data = pd.DataFrame, pd.read_csv(summary_file)  # type: ignore
        return data

    def get_status(self) -> dict[str, int]:
        """
        Get the status of the experiment configurations.

        Returns:
            A dictionary with keys 'todo', 'done', 'failed' and their counts
        """
        state = self.load_state()
        assert state is not None
        statuses = state.configs.values()
        num_todo = sum(1 for c in statuses if c.status == "todo")
        num_done = sum(1 for c in statuses if c.status == "done")
        num_failed = sum(1 for c in statuses if c.status == "failed")
        return {"todo": num_todo, "done": num_done, "failed": num_failed}

    def run_cli(self):
        """
        Run the experiment as a CLI application.
        """
        fire.Fire(ExperimentCLI(self))  # type: ignore


EXPORTED_BUDGET_FIELDS = [
    "num_completions",
    "num_requests",
    "input_tokens",
    "cached_input_tokens",
    "output_tokens",
    "price",
]


def results_summary(
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
    experiment: _ExperimentFun[Config],
    config_name: str,
    config_dir: Path,
    config: Config,
    cache_requests: bool,
    export_raw_trace: bool,
    export_log: bool,
    export_browsable_trace: bool,
) -> tuple[str, bool]:
    cache_dir = None
    if cache_requests:
        cache_dir = config_dir / CACHE_DIR
        if cache_dir.exists():
            shutil.rmtree(cache_dir, ignore_errors=True)
    for f in (STATUS_FILE, RESULT_FILE, LOG_FILE):
        file_path = config_dir / f
        if file_path.exists():
            file_path.unlink(missing_ok=True)
    cmdargs = experiment(config)
    if cache_requests:
        cmdargs.requests_cache = CACHE_DIR
    cmdargs.requests_cache_mode = "create"
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
    return json.dumps(python)


class ExperimentCLI:
    def __init__(self, experiment: Experiment[Any]):
        self.experiment = experiment

    def __call__(self):
        self.run()

    def run(
        self,
        max_workers: int = 1,
        retry_errors: bool = False,
        cache: bool = True,
        verbose_output: bool = False,
    ):
        self.experiment.cache_requests = cache
        self.experiment.export_raw_trace = verbose_output
        self.experiment.export_browsable_trace = verbose_output
        self.experiment.export_log = True

        self.experiment.load()
        if retry_errors:
            self.experiment.mark_errors_as_todos()
        self.experiment.resume(max_workers=max_workers)

    def status(self):
        status_counts = self.experiment.get_status()
        print(
            f"Experiment '{self.experiment.name}':\n"
            f"  - {status_counts['todo']} configurations to do\n"
            f"  - {status_counts['done']} configurations done\n"
            f"  - {status_counts['failed']} configurations failed"
        )

    def replay(self, config: str | None = None):
        self.experiment.load()
        if config is None:
            self.experiment.replay_all_configs()
        else:
            self.experiment.replay_config_by_name(config)


def quick_experiment[Config](
    fun: _ExperimentFun[Config],
    configs: Sequence[Config],
    *,
    name: str,
    workspace_root: Path,
    modules: Sequence[str],
    demo_files: Sequence[str],
    output_dir: Path | str | None = None,
    strategy_dirs: Sequence[Path | str] | None = None,
    demo_dirs: Sequence[Path | str] | None = None,
    data_dirs: Sequence[Path | str] | None = None,
    prompt_dirs: Sequence[Path | str] | None = None,
    config_naming: Callable[[Config, uuid.UUID], str] | None = None,
    config_type: type[Config] | None = None,
) -> Experiment[Config]:
    if strategy_dirs is None:
        strategy_dirs = ["."]
    if demo_dirs is None:
        demo_dirs = ["."]
    if data_dirs is None:
        data_dirs = ["data"]
    if prompt_dirs is None:
        prompt_dirs = ["prompts"]
    if output_dir is None:
        output_dir = Path("experiments") / "output"
    context = CommandExecutionContext(
        base=analysis.DemoExecutionContext(
            strategy_dirs=[workspace_root / d for d in strategy_dirs],
            modules=modules,
        ),
        demo_files=[workspace_root / (f + ".demo.yaml") for f in demo_files],
        prompt_dirs=[workspace_root / d for d in prompt_dirs],
        data_dirs=[workspace_root / d for d in data_dirs],
        requests_cache_dir=None,
        result_refresh_period=None,
        status_refresh_period=None,
    )
    if config_type is None:
        assert configs, "Empty list of configurations."
        config_type = type(configs[0])

    return Experiment(
        name=name,
        dir=workspace_root / output_dir / name,
        context=context,
        experiment=fun,
        config_type=config_type,
        configs=configs,
        config_naming=config_naming,
    )
