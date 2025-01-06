"""
Demonstration Interpreter.
"""

import importlib
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import delphyne.core as dp
from delphyne.utils import typing as tp

#####
##### Environment Execution Context
#####


@dataclass
class ModuleNotFound(Exception):
    module_name: str


@dataclass
class ObjectNotFound(Exception):
    object_name: str


@dataclass
class StrategyLoadingError(Exception):
    message: str


@dataclass(frozen=True)
class DemoExecutionContext:
    strategy_dirs: Sequence[Path]
    modules: Sequence[str]


class ObjectLoader:
    def __init__(self, ctx: DemoExecutionContext, reload: bool = True):
        """
        Raises `ModuleNotFound`.
        """
        self.ctx = ctx
        self.modules: list[Any] = []
        with _append_path(self.ctx.strategy_dirs):
            for module_name in self.modules:
                try:
                    module = __import__(module_name)
                    if reload:
                        module = importlib.reload(module)
                    self.modules.append(module)
                except AttributeError:
                    raise ModuleNotFound(module_name)

    def find_object(self, name: str) -> Any:
        for module in self.modules:
            if hasattr(module, name):
                return getattr(module, name)
        raise ObjectNotFound(name)

    def load_strategy_instance(
        self, name: str, args: dict[str, Any]
    ) -> dp.StrategyComp[Any, Any, Any]:
        f = self.find_object(name)
        try:
            args = tp.parse_function_args(f, args)
            return f(**args)
        except Exception as e:
            raise StrategyLoadingError(str(e))


@contextmanager
def _append_path(paths: Sequence[Path]):
    sys.path = [str(p) for p in paths] + sys.path
    yield
    sys.path = sys.path[len(paths) :]
