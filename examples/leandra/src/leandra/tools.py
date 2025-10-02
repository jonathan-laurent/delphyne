"""
Simple API to communicate with Lean and related tools
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lean_interact as li  # type: ignore
import lean_interact.interface as li_intf  # type: ignore

DEFAULT_MEMORY_HARD_LIMIT_MB = 4096
DEFAULT_TIMEOUT_IN_SECONDS = 5.0


#####
##### Global Server Initialization
#####


@dataclass
class _LeanServer:
    server: li.AutoLeanServer
    env: int | None


_global_lean_server: _LeanServer | None = None


def _default_lean_version() -> str:
    workspace = Path(__file__).absolute().parent.parent.parent
    toolchain_file = workspace / "benchmarks" / "minif2f" / "lean-toolchain"
    return toolchain_file.read_text().strip()


def lean_server_config(
    *,
    repo_path: Path | None,
    memory_hard_limit_mb: int | None,
) -> li.LeanREPLConfig:
    """
    Obtain a REPL configuration object, which can then be shared across
    workers and passed as an argument to
    `init_global_lean_server_with_config`.

    Arguments:
        repo_path: The path to the root of a Lean project that imports
            all the necessary libraries. If not provided,
            `lean_interact` will be tasked to generate a temporary
            project on the fly, which might be slower.
        memory_hard_limit_mb: The memory limit in megabytes for each
            server worker.
    """

    if memory_hard_limit_mb is None:
        memory_hard_limit_mb = DEFAULT_MEMORY_HARD_LIMIT_MB
    if repo_path is None:
        project = li.TempRequireProject(
            lean_version=_default_lean_version(), require="mathlib"
        )
    else:
        project = li.LocalProject(directory=repo_path, auto_build=False)
    return li.LeanREPLConfig(
        project=project,
        memory_hard_limit_mb=memory_hard_limit_mb,
        verbose=True,
    )


def init_global_lean_server_with_config(
    config: li.LeanREPLConfig,
    init_commands: Sequence[str],
) -> None:
    """
    Initialize the global Lean server.

    When using multiprocessing, this function should be called in each
    worker process.
    """
    if _global_lean_server is not None:
        return
    server = li.AutoLeanServer(config)
    env = None
    for cmd in init_commands:
        res = server.run(
            li.Command(cmd=cmd, env=env),
            add_to_session_cache=True,
        )
        if isinstance(res, li_intf.LeanError):
            raise RuntimeError(
                f"Failed to run init command '{cmd}': {res.message}"
            )
        env = res.env
    global _global_lean_server
    _global_lean_server = _LeanServer(server=server, env=env)


def init_global_lean_server(
    *,
    repo_path: Path | None,
    memory_hard_limit_mb: int | None,
    init_commands: Sequence[str],
) -> None:
    """
    Shortcut for configuring an REPL and launching a single server.
    """
    if _global_lean_server is not None:
        return
    config = lean_server_config(
        repo_path=repo_path,
        memory_hard_limit_mb=memory_hard_limit_mb,
    )
    init_global_lean_server_with_config(config, init_commands)


def _get_global_server() -> _LeanServer:
    global _global_lean_server
    if _global_lean_server is None:
        raise RuntimeError("The Lean server was not initialized.")
    return _global_lean_server


#####
##### Executing Commands
#####


def run_lean_command(
    command: str, *, timeout_in_seconds: float | None = None
) -> li_intf.BaseREPLResponse:
    """
    Run a Lean command.

    Attributes:
        command: The Lean command to run.
        timeout_in_seconds: The maximum time to wait for a response. If the
            timeout is reached, a response with a timeout error message
            will be returned. If `None`, a default timeout is used.
    """
    if timeout_in_seconds is None:
        timeout_in_seconds = DEFAULT_TIMEOUT_IN_SECONDS
    server = _get_global_server()
    try:
        cmd = li.Command(cmd=command, env=server.env)
        resp = server.server.run(cmd, timeout=1e-4)
        if isinstance(resp, li_intf.LeanError):
            raise RuntimeError(
                f"Failed to run command:\n\n{command}\n\nError: {resp.message}"
            )
        return resp
    except TimeoutError:
        dummy_pos = li_intf.Pos(line=0, column=0)
        message = li_intf.Message(
            start_pos=dummy_pos,
            end_pos=dummy_pos,
            severity="error",
            data="Timeout",
        )
        return li_intf.BaseREPLResponse(messages=[message], sorries=[])
