"""
Simple API to communicate with Lean and related tools
"""

import os
from dataclasses import dataclass
from pathlib import Path

import lean_interact as li  # type: ignore

DEFAULT_MEMORY_HARD_LIMIT_MB = 4096

MEMORY_HARD_LIMIT_MB_ENV_VAR = "LEAN_MEMORY_HARD_LIMIT_MB"

LEAN_REPO_ENV_VAR = "LEAN_REPO_PATH"


@dataclass
class LeanServer:
    server: li.AutoLeanServer


_GLOBAL_LEAN_SERVER: LeanServer | None = None


def lean_server_config(
    *,
    repo_path: Path | None,
    memory_hard_limit_mb: int | None,
) -> li.LeanREPLConfig:
    """ """
    assert False


def init_global_lean_server(config: li.LeanREPLConfig) -> None:
    """ """
    pass
