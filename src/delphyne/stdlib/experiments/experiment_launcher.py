"""
Utilities to launch experiments in Delphyne.


An experiment is associated a directory.
"""

import shutil
import uuid
from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol, cast

import yaml

import delphyne.stdlib.commands as cmd
import delphyne.stdlib.models as md
from delphyne.stdlib.tasks import CommandExecutionContext, run_command
from delphyne.utils.typing import pydantic_dump, pydantic_load

type _Config = dict[str, Any]


type _ModelWrapper = Callable[[md.LLM], md.LLM]


EXPERIMENT_STATE_FILE = "experiment.yaml"
STATUS_FILE = "statuses.txt"
RESULT_FILE = "result.yaml"
LOG_FILE = "log.txt"
EXCEPTION_FILE = "exception.txt"
CACHE_DIR = "llm_cache"


@dataclass
class ConfigInfo:
    params: dict[str, Any]
    status: Literal["todo", "done", "failed"]


@dataclass
class ExperimentState:
    name: str | None
    description: str | None
    configs: dict[str, ConfigInfo]


class _ExperimentFun(Protocol):
    def __call__(
        self, cache_dir: Path, **args: Any
    ) -> cmd.RunStrategyArgs: ...


@dataclass
class Experiment:
    dir: Path
    context: CommandExecutionContext
    experiment: _ExperimentFun
    configs: Sequence[_Config] | None = None
    name: str | None = None
    description: str | None = None
    config_naming: Callable[[_Config, uuid.UUID], str] | None = None

    def __post_init__(self):
        if not self.dir_exists():
            # If we create the experiment for the first time
            self.dir.mkdir(parents=True, exist_ok=True)
            state = self.initial_state()
            self.save_state(state)

    def config_dir(self, config_name: str) -> Path:
        dir = self.dir / config_name
        dir.mkdir(parents=True, exist_ok=True)
        return dir

    def initial_state(self) -> ExperimentState:
        assert self.configs is not None
        configs: dict[str, ConfigInfo] = {}
        for c in self.configs:
            id = uuid.uuid4()
            if self.config_naming is not None:
                name = self.config_naming(c, id)
            else:
                name = str(id)
            configs[name] = ConfigInfo(c, status="todo")
        return ExperimentState(self.name, self.description, configs)

    def dir_exists(self) -> bool:
        return self.dir.exists() and self.dir.is_dir()

    def load_state(self) -> ExperimentState | None:
        with open(self.dir / EXPERIMENT_STATE_FILE, "r") as f:
            parsed = yaml.safe_load(f)
            return pydantic_load(ExperimentState, parsed)

    def save_state(self, state: ExperimentState) -> None:
        with open(self.dir / EXPERIMENT_STATE_FILE, "w") as f:
            yaml.safe_dump(pydantic_dump(ExperimentState, state), f)

    def resume(self, max_workers: int = 1, log_progress: bool = True):
        state = self.load_state()
        assert state is not None
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    _run_config,
                    self.context,
                    self.experiment,
                    name,
                    self.config_dir(name),
                    info.params,
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

    def mark_errors_as_todos(self):
        state = self.load_state()
        assert state is not None
        for _, info in state.configs.items():
            if info.status == "failed":
                info.status = "todo"
        self.save_state(state)


def _print_progress(state: ExperimentState) -> None:
    num_done = sum(1 for c in state.configs.values() if c.status != "todo")
    num_failed = sum(1 for c in state.configs.values() if c.status == "failed")
    num_total = len(state.configs)
    msg = f"\rDone: {num_done} / {num_total}, Failed: {num_failed}"
    print(msg + 40 * " ", end="")


def _run_config(
    context: CommandExecutionContext,
    experiment: _ExperimentFun,
    config_name: str,
    config_dir: Path,
    config: _Config,
) -> tuple[str, bool]:
    cache_dir = config_dir / CACHE_DIR
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    for f in (STATUS_FILE, RESULT_FILE, LOG_FILE):
        file_path = config_dir / f
        if file_path.exists():
            file_path.unlink(missing_ok=True)
    cmdargs = cast(Any, experiment(cache_dir, **config))
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
        with open(config_dir / EXCEPTION_FILE, "w") as f:
            import traceback

            traceback.print_exc(file=f)
        success = False
    return (config_name, success)
