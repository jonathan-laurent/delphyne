"""
Utilities to launch experiments in Delphyne.


An experiment is associated a directory.
"""

import json
import shutil
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import pandas as pd  # type: ignore
import yaml

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
    def __call__(
        self, cache_dir: Path | None, config: Config
    ) -> cmd.RunStrategyArgs: ...


@dataclass
class Experiment[Config]:
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
            c for c, i in state.configs.items()
            if _config_unique_repr(i.params) not in in_config]
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
            yaml.safe_dump(pydantic_dump(self.state_type(), state), f)

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
            for future in as_completed(futures):
                name, success = future.result()
                state.configs[name].status = "done" if success else "failed"
                self.save_state(state)
                if log_progress:
                    _print_progress(state)
            all_successes = all(
                info.status == "done" for info in state.configs.values()
            )
            if all_successes:
                print("\nExperiment successful.\nProducing summary file...")
                self.save_summary()
            else:
                print("\nWarning: some configurations failed.")

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

    def replay_config(self, config: Config) -> None:
        """
        Replay a configuration, reusing the cache if it exists.

        This way, one can debug the execution of an experiment after the
        fact, without any LLMs being called.
        """
        state = self.load_state()
        assert state is not None
        config_name = self.existing_config_name(config)
        assert config_name is not None
        info = state.configs[config_name]
        assert info.status == "done"
        dir = self.config_dir(config_name)
        cmdargs = self.experiment(dir / CACHE_DIR, info.params)
        run_command(
            cmd.run_strategy,
            cmdargs,
            self.context,
            dump_statuses=None,
            dump_result=None,
            dump_log=None,
        )

    def save_summary(self):
        """
        Save a summary of the results in a CSV file.
        """

        data = results_summary(self.dir)
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


def results_summary(exp_dir: Path) -> Sequence[dict[str, Any]]:
    # Load the experiment state as a python dict
    state_file = exp_dir / EXPERIMENT_STATE_FILE
    with open(state_file, "r") as f:
        parsed = yaml.safe_load(f)
    res: list[dict[str, Any]] = []
    for name, info in parsed["configs"].items():
        params = info["params"]
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
        num_completions = spent.get("num_completions", 0)
        num_requests = spent.get("num_requests", 0)
        assert isinstance(price, float)
        entry = params | {
            "success": success,
            "price": price,
            "num_completions": num_completions,
            "num_requests": num_requests,
        }
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
    # TODO: TEMPORARY: we send the path as a string despite the type
    # annotation because to avoid YAML serialization issues.
    cmdargs = experiment(str(cache_dir), config)  # type: ignore
    cmdargs.export_browsable_trace = export_browsable_trace
    cmdargs.export_log = export_log
    cmdargs.export_raw_trace = export_raw_trace
    try:
        run_command(
            cmd.run_strategy,
            cmdargs,
            context,
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
