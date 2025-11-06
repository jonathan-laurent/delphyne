"""
Execution Contexts
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

import delphyne.core as dp
import delphyne.stdlib.embeddings as em
import delphyne.stdlib.environments as en
import delphyne.stdlib.models as md
import delphyne.utils.typing as ty
from delphyne.analysis.object_loaders import (
    ObjectLoader,
    ObjectLoaderInitializer,
)
from delphyne.stdlib.globals import stdlib_globals

WORKSPACE_FILE = "delphyne.yaml"

DEFAULT_STRATEGY_DIRS = (Path("."),)
DEFAULT_PROMPTS_DIRS = (Path("prompts"),)
DEFAULT_DATA_DIRS = (Path("data"),)
DEFAULT_GLOBAL_EMBEDDINGS_CACHE_FILE = Path("embeddings.cache.h5")


@dataclass(frozen=True, kw_only=True)
class ExecutionContext:
    """
    Context information available to all commands.

    This information is usually specified in `delphyne.yaml` project
    files, and also locally in `@config` blocks. All values have
    defaults. All paths are usually expressed relative to a single
    workspace root directory, and can be made absolute using the
    `with_root` method.

    Parameters:
        strategy_dirs: A sequence of directories in which strategy
            modules can be found.
        modules: A sequence of module in which Python object identifiers
            can be resolved (module names can feature `.`).
        demo_files: A sequence of demonstration files (either including or
            excluding the `*.demo.yaml` extension).
        prompt_dirs: A sequence of directories in which to look for
            prompt templates.
        data_dirs: A sequence of directories in which to look for data
            files.
        cache_root: The directory in which to store all request cache
            subdirectories.
        global_embeddings_cache_file: Global cache file that stores
            common embeddings (e.g. embeddings of examples).
        init: A sequence of initialization functions to call
            before any object is loaded. Each element specifies a
            qualified function name, or a pair of a qualified function
            name and of a dictionary of arguments to pass. Each
            initializer function is called at most once per Python
            process (subsequent calls with possibly different arguments
            are ignored). For all string arguments, the "%workspace"
            substring is replaced by the path of the workspace root
            directory (without trailing slash).
        result_refresh_period: The period in seconds at which the
            current result is computed and communicated to the UI (e.g.,
            the period at which the current trace is exported when
            running oracular programs). If `None`, the result is never
            refreshed (until the command terminates).
        status_refresh_period: The period in seconds at which the
            current status message is communicated to the UI. If `None`,
            the status is never refreshed (until the command
            terminates).
        workspace_root: The root directory of the workspace. This value
            is not meant to be provided in configuration files. Rather,
            it is set when using the `with_root` method. This setting is
            useful for interpreting some relative paths in demonstration
            and command files (e.g. `using` directives).

    !!! info "Local conguration blocks"
        Demonstration and command files can override some configuration
        information from the `delphyne.yaml` file by featuring a comment
        block such as:

        ```yaml
        # @config
        # modules: ["my_strategy_module"]
        # demo_files: ["demo.yaml"]
        # @end
        ```

        The comment block must be placed at the start of the file,
        possibly after other comments.
    """

    strategy_dirs: Sequence[Path] = DEFAULT_STRATEGY_DIRS
    modules: Sequence[str] = ()
    demo_files: Sequence[Path] = ()
    prompt_dirs: Sequence[Path] = DEFAULT_PROMPTS_DIRS
    data_dirs: Sequence[Path] = DEFAULT_DATA_DIRS
    cache_root: Path | None = None
    global_embeddings_cache_file: Path = DEFAULT_GLOBAL_EMBEDDINGS_CACHE_FILE
    init: Sequence[str | ObjectLoaderInitializer] = ()
    result_refresh_period: float | None = None
    status_refresh_period: float | None = None
    workspace_root: Path | None = None

    def with_root(self, root: Path) -> "ExecutionContext":
        """
        Make all paths absolute given a path to the workspace root.
        """

        def _expand_initializer(
            init: str | ObjectLoaderInitializer,
        ) -> str | ObjectLoaderInitializer:
            if isinstance(init, str):
                return init
            else:
                args = {
                    k: v.replace("%workspace", str(root))
                    if isinstance(v, str)
                    else v
                    for k, v in init.args.items()
                }
                return ObjectLoaderInitializer(init.function, args)

        return ExecutionContext(
            modules=self.modules,
            demo_files=[root / f for f in self.demo_files],
            strategy_dirs=[root / d for d in self.strategy_dirs],
            prompt_dirs=[root / d for d in self.prompt_dirs],
            data_dirs=[root / d for d in self.data_dirs],
            cache_root=None if self.cache_root is None else self.cache_root,
            global_embeddings_cache_file=(
                root / self.global_embeddings_cache_file
            ),
            init=[_expand_initializer(i) for i in self.init],
            result_refresh_period=self.result_refresh_period,
            status_refresh_period=self.status_refresh_period,
            workspace_root=root,
        )

    def object_loader(
        self,
        *,
        extra_objects: dict[str, Any] | None,
    ) -> ObjectLoader:
        return ObjectLoader(
            strategy_dirs=self.strategy_dirs,
            modules=self.modules,
            extra_objects=extra_objects,
            initializers=self.init,
        )

    def policy_env(
        self,
        cache: md.LLMCache | None = None,
        embeddings_cache: em.EmbeddingsCache | None = None,
        override_answers: dp.AnswerDatabase | None = None,
        log_level: dp.LogLevel = "info",
        log_long_computations: tuple[dp.LogLevel, float] | None = None,
        random_seed: int = 0,
    ):
        return en.PolicyEnv(
            object_loader=self.object_loader(extra_objects=stdlib_globals()),
            prompt_dirs=self.prompt_dirs,
            demonstration_files=self.demo_files,
            data_dirs=self.data_dirs,
            cache=cache,
            embeddings_cache=embeddings_cache,
            global_embeddings_cache_file=self.global_embeddings_cache_file,
            override_answers=override_answers,
            log_level=log_level,
            log_long_computations=log_long_computations,
            random_seed=random_seed,
        )


#####
##### Loading execution contexts
#####


def load_execution_context(
    workspace_dir: Path, local: Path | None = None
) -> ExecutionContext:
    """
    Load an execution context from a workspace directory.

    If this directory contains a `delphyne.yaml` file, load it.
    Otherwise, load the default execution context.

    Attributes:
        workspace_dir: The root of the workspace directory.
        local: An optional path to a local file
            (e.g., a demonstration or command file) from which to
            extract a local configuration block to override the global
            configuration
    """
    config_path = workspace_dir / WORKSPACE_FILE
    if config_path.exists():
        with open(config_path, "r") as f:
            data: Any = yaml.safe_load(f) or {}
    else:
        data = {}
    # If local_config_from is provided, merge local config
    if local is not None:
        with open(local, "r") as f:
            document = f.read()
        local_block = _extract_config_block(document)
        if local_block is not None:
            local_data: Any = yaml.safe_load(local_block) or {}
            assert isinstance(local_data, dict)
            data = data | local_data
    cex = ty.pydantic_load(ExecutionContext, data)
    return cex.with_root(workspace_dir)


def _extract_config_block(document: str) -> str | None:
    lines = document.split("\n")
    config_start_index = -1
    config_end_index = -1

    # Find the start of the config block
    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # If we encounter a non-comment, non-blank line before finding
        # @config, return None
        if line_stripped and not line_stripped.startswith("#"):
            return None
        if line_stripped == "# @config":
            config_start_index = i
            break
    if config_start_index == -1:
        return None

    # Find the end of the config block and validate content
    for i in range(config_start_index + 1, len(lines)):
        line_stripped = lines[i].strip()
        if line_stripped == "# @end":
            config_end_index = i
            break
        if line_stripped and not line_stripped.startswith("#"):
            return None
    if config_end_index == -1:
        return None

    # Extract the content between @config and @end
    config_lines: list[str] = []
    for i in range(config_start_index + 1, config_end_index):
        line = lines[i]
        if line.startswith("# "):
            config_lines.append(line[2:])
        elif line.strip() == "#":
            config_lines.append("")
        else:
            config_lines.append(line)
    return "\n".join(config_lines)


def surrounding_workspace_dir(starting_dir: Path) -> Path | None:
    """
    Find the workspace directory by looking for the delphyne.yaml file.
    """
    current_dir = starting_dir.resolve()
    while current_dir != current_dir.parent:
        if (current_dir / WORKSPACE_FILE).exists():
            return current_dir
        current_dir = current_dir.parent
    return None


def workspace_execution_context(
    current_dir: Path | str,
) -> ExecutionContext:
    """
    Load the execution context by looking for a `delphyne.yaml` file in
    a parent directory.

    This is a convenience wrapper that combines
    `surrounding_workspace_dir` and `load_execution_context`.
    """
    if isinstance(current_dir, str):
        current_dir = Path(current_dir)
    workspace_dir = surrounding_workspace_dir(current_dir)
    if workspace_dir is None:
        raise ValueError("No delphyne.yaml found in parent directories.")
    return load_execution_context(workspace_dir)
