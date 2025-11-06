"""
Simple API to communicate with Lean and related tools
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import lean_interact as li
import lean_interact.interface as li_intf

DEFAULT_MEMORY_HARD_LIMIT_MB = 8192
DEFAULT_TIMEOUT_IN_SECONDS = 5.0
DEBUG_MODE = True

#####
##### Global Server Initialization
#####


def dbg(s: str) -> None:
    if DEBUG_MODE:
        print("[leandra.tools]", s, flush=True)


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

    dbg("Creating Lean REPL config...")
    if memory_hard_limit_mb is None:
        memory_hard_limit_mb = DEFAULT_MEMORY_HARD_LIMIT_MB
    if repo_path is None:
        project = li.TempRequireProject(
            lean_version=_default_lean_version(), require="mathlib"
        )
    else:
        project = li.LocalProject(directory=repo_path, auto_build=False)
    config = li.LeanREPLConfig(
        project=project,
        memory_hard_limit_mb=memory_hard_limit_mb,
        verbose=True,
    )
    dbg("Lean REPL config created.")
    return config


def init_global_lean_server_with_config(
    config: li.LeanREPLConfig,
    init_commands: Sequence[str],
) -> None:
    """
    Initialize the global Lean server.

    When using multiprocessing, this function should be called in each
    worker process.
    """
    global _global_lean_server
    if _global_lean_server is not None:
        return
    dbg("Creating Lean Server...")
    server = li.AutoLeanServer(config)
    dbg("Lean server created.")
    env = None
    # We do not want the user to mistakenly pass a string here.
    assert isinstance(init_commands, (list, tuple)) and all(
        isinstance(cmd, str) for cmd in init_commands
    ), "Invalid init_commands argument when initializing Lean server."
    for cmd in init_commands:
        dbg(f"Run initialization command:\n{cmd}")
        res = server.run(
            li.Command(cmd=cmd, env=env),
            add_to_session_cache=True,
        )
        if isinstance(res, li_intf.LeanError):
            raise RuntimeError(
                f"Failed to run init command '{cmd}': {res.message}"
            )
        env = res.env
    _global_lean_server = _LeanServer(server=server, env=env)


def init_global_lean_server(
    *,
    repo_path: Path | None,
    init_commands: Sequence[str],
    memory_hard_limit_mb: int | None = None,
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


@dataclass
class LeanResponse:
    messages: list[li_intf.Message]
    sorries: list[li_intf.Sorry]

    @staticmethod
    def make(resp: li_intf.BaseREPLResponse) -> "LeanResponse":
        return LeanResponse(messages=resp.messages, sorries=resp.sorries)


def run_lean_command(
    command: str, *, timeout_in_seconds: float | None = None
) -> LeanResponse:
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
        dbg(f"Running Lean command:\n{command}")
        cmd = li.Command(cmd=command, env=server.env)
        resp = server.server.run(cmd, timeout=timeout_in_seconds)
        if isinstance(resp, li_intf.LeanError):
            raise RuntimeError(
                f"Failed to run command:\n\n{command}\n\nError: {resp.message}"
            )
        return LeanResponse.make(resp)
    except TimeoutError:
        dummy_pos = li_intf.Pos(line=0, column=0)
        message = li_intf.Message(
            start_pos=dummy_pos,
            end_pos=dummy_pos,
            severity="error",
            data="Timeout",
        )
        return LeanResponse(messages=[message], sorries=[])
