"""
Object Loaders
"""

import importlib
import sys
import threading
from collections import defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import delphyne.core as dp
from delphyne.utils import typing as tp


@dataclass
class ModuleNotFound(Exception):
    """
    Raised by `ObjectLoader` when a module is not found.
    """

    module_name: str


@dataclass
class ObjectNotFound(Exception):
    """
    Raised by `ObjectLoader` when an object cannot be found.
    """

    object_name: str


@dataclass
class StrategyLoadingError(Exception):
    """
    Raised by `ObjectLoader` when a strategy instance cannot be loaded.
    """

    message: str


@dataclass(frozen=True)
class AmbiguousObjectIdentifier(Exception):
    """
    Raised when attempting to load an object with an ambiguous name.

    Attributes:
        identifier: the ambiguous identifier.
        modules: a list of modules where different objects with the same
            identifier were found
    """

    identifier: str
    modules: Sequence[str]


_GLOBAL_OBJECT_LOADER_LOCK = threading.Lock()
"""
Global lock that ensures that instances of `ObjectLoader` are never
initialized concurrently.
"""


_GLOBAL_OBJECT_LOADER_EXECUTED_INITIALIZERS: set[int] = set()
"""
The set of ids of initializers that have already been executed, so that
no initializer is executed several times.
"""


@dataclass
class ObjectLoaderInitializer:
    """
    Specification of a function to be called upon creation of an object
    loader.
    """

    function: str
    args: dict[str, Any]


class ObjectLoader:
    """
    Utility class for loading Python objects.

    Demonstration and command files may refer to Python identifiers that
    need to be resolved. This is done relative to a list of directories
    to be added to `sys.path`, along with a list of modules.

    An exception is raised if an object with the requested identifier
    can be found in several modules.
    """

    def __init__(
        self,
        *,
        strategy_dirs: Sequence[Path],
        modules: Sequence[str],
        extra_objects: dict[str, object] | None = None,
        initializers: Sequence[str | ObjectLoaderInitializer] = (),
    ):
        """
        Attributes:
            strategy_dirs: A list of directories in which strategy
                modules can be found, to be added to `sys.path`.
            modules: A list of modules in which python object
                identifiers should be resolved. Modules can be part of
                packages and so their name may feature `.`.
            extra_objects: Additional objects that can be resolved by
                name (with higher precedence).
            initializers: A sequence of initialization functions to call
                before any object is loaded. Each element specifies a
                qualified function name, or a pair of a qualified
                function name and of a dictionary of arguments to pass.
                Each initializer function is called at most once per
                Python process (subsequent calls with possibly different
                arguments are ignored).

        Raises:
            ModuleNotFound: a module could not be found.
        """
        self.extra_objects = extra_objects if extra_objects is not None else {}
        self.modules: list[Any] = []
        with _GLOBAL_OBJECT_LOADER_LOCK:
            with _append_path(strategy_dirs):
                for module_name in modules:
                    try:
                        module = importlib.import_module(module_name)
                        self.modules.append(module)
                    except AttributeError:
                        raise ModuleNotFound(module_name)
            for initializer in initializers:
                match initializer:
                    case str() as name:
                        f = self.find_object(name)
                        args = {}
                    case ObjectLoaderInitializer(name, args):
                        f = self.find_object(name)
                if not callable(f):
                    raise TypeError(f"Initializer {name} is not callable.")
                if id(f) not in _GLOBAL_OBJECT_LOADER_EXECUTED_INITIALIZERS:
                    f(**args)
                    # We only count the initializer as executed after a
                    # successful call. This way, if the initializer
                    # raises an exception, the parent command can be run
                    # again after fixing the issue (e.g., modifying
                    # `delphyne.yaml`).
                    _GLOBAL_OBJECT_LOADER_EXECUTED_INITIALIZERS.add(id(f))

    @staticmethod
    def trivial() -> "ObjectLoader":
        """
        Create a trivial object loader that always fails at loading
        objects.
        """
        return ObjectLoader(strategy_dirs=[], modules=[])

    def find_object(self, name: str) -> Any:
        """
        Find an object with a given name.

        If the name is unqualified (it features no `.`), one attempts to
        find the object in every registered module in order. If the name
        is qualified, one looks at the specified registered module.

        Raises:
            ObjectNotFound: The object could not be found.
            AmbiguousObjectIdentifier: The object name is ambiguous,
                i.e. it is found in several modules.
        """
        if name in self.extra_objects:
            return self.extra_objects[name]
        comps = name.split(".")
        assert comps
        if len(comps) == 1:
            # unqualified name
            cands: list[object] = []
            modules_with_id: dict[int, list[str]] = defaultdict(list)
            for module in self.modules:
                if hasattr(module, name):
                    obj = getattr(module, name)
                    modules_with_id[id(obj)].append(module)
                    cands.append(obj)
            if len(modules_with_id) > 1:
                ambiguous = [ms[0] for ms in modules_with_id.values()]
                raise AmbiguousObjectIdentifier(name, ambiguous)
            if cands:
                return cands[0]
        else:
            # qualified name
            module = ".".join(comps[:-1])
            attr = comps[-1]
            if hasattr(module, attr):
                return getattr(module, attr)
        raise ObjectNotFound(name)

    def load_and_call_function(self, name: str, args: dict[str, Any]) -> Any:
        """
        Load and call a function by wrapping a call to `find_object`.
        """
        f = self.find_object(name)
        args = tp.parse_function_args(f, args)
        return f(**args)

    def load_strategy_instance(
        self, name: str, args: dict[str, Any]
    ) -> dp.StrategyComp[Any, Any, Any]:
        """
        Load and instantiate a strategy function with given arguments.

        Raises:
            ObjectNotFound: If the strategy function cannot be found.
            AmbiguousObjectIdentifier: If an ambiguous name is given.
            StrategyLoadingError: If the object is not a strategy function
                or if the arguments are invalid.
        """
        f = self.find_object(name)
        try:
            args = tp.parse_function_args(f, args)
            comp = f(**args)
            assert isinstance(comp, dp.StrategyComp), (
                f"Object {name} is not a strategy function."
                + " Did you forget to use the @strategy decorator?"
            )
            return cast(Any, comp)
        except Exception as e:
            raise StrategyLoadingError(str(e))

    def load_query(
        self, name: str, args: dict[str, Any]
    ) -> dp.AbstractQuery[Any]:
        """
        Load a query by name and instantiate it with given arguments.

        Raises:
            ObjectNotFound: if the query cannot be found.
            AmbiguousObjectIdentifier: if an ambiguous name is given.
            AssertionError: if the object is not a query.
        """
        obj = self.find_object(name)
        assert issubclass(obj, dp.AbstractQuery), (
            f"Object {name} is not a query type."
        )
        q = cast(type[dp.AbstractQuery[Any]], obj)
        return q.parse_instance(args)


@contextmanager
def _append_path(paths: Sequence[Path]):
    sys.path = [str(p) for p in paths] + sys.path
    try:
        yield
    finally:
        sys.path = sys.path[len(paths) :]
