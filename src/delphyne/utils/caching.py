"""
Utilities for memoizing function calls.
"""

import functools
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import TypeAdapter

from delphyne.utils.pretty_yaml import pretty_yaml

type CacheMode = Literal["read_write", "off", "create", "replay"]
"""
Caching mode:

- `off`: the cache is disabled.
- `read_write`: values can be read and written to the cache (no extra
      check is made).
- `create`: the cache is used in write-only mode, and an exception is
      raided if a cached value already exists.
- `replay`: all requests must hit the cache or an exception is raised.
"""


@dataclass(frozen=True)
class Cache[P, T]:
    """
    A cache backed by a dictionary, which can be used to memoize
    function calls.
    """

    dict: dict[P, T]
    mode: CacheMode

    def __call__(self, func: Callable[[P], T]) -> Callable[[P], T]:
        """
        Decorate a function to use the cache.
        """

        @functools.wraps(func)
        def cached_func(arg: P) -> T:
            if self.mode == "off":
                return func(arg)
            if arg in self.dict:
                assert self.mode != "create", "Cache entry already exists."
                return self.dict[arg]
            assert self.mode != "replay", (
                f"Cache entry not found for:\n\n {arg}"
            )
            ret = func(arg)
            self.dict[arg] = ret
            return ret

        return cached_func

    def batched(
        self, func: Callable[[Sequence[P]], Sequence[T]]
    ) -> Callable[[Sequence[P]], Sequence[T]]:
        """
        Decorate a **batched** evaluation function to use the cache.

        Whenever the resulting function is called, some elements are
        taken from the cache while others are computed by calling
        `func`.

        !!! note
            The case where some batch elements are identical is
            consistently handled, in the sense that the same cached
            answer is returned for all of them.
        """

        @functools.wraps(func)
        def cached_func(args: Sequence[P]) -> Sequence[T]:
            if self.mode == "off":
                return func(args)
            n = len(args)
            cached_already: set[int] = set()
            for i in range(n):
                arg = args[i]
                if arg in self.dict:
                    assert self.mode != "create", "Cache entry already exists."
                    cached_already.add(i)
            to_compute = [i for i in range(n) if i not in cached_already]
            if to_compute:
                assert self.mode != "replay", (
                    f"Cache entry not found for:\n\n{args[to_compute[0]]}"
                )
            computed = func([args[j] for j in to_compute])
            for i, v in zip(to_compute, computed):
                self.dict[args[i]] = v
            return [self.dict[a] for a in args]

        return cached_func


@contextmanager
def load_cache(
    file: Path, *, input_type: Any, output_type: Any, mode: CacheMode
):
    """
    Load a cache from a YAML file on disk.
    """
    assoc_type = AssocList[input_type, output_type]
    assoc_adapter = TypeAdapter[AssocList[Any, Any]](assoc_type)
    # Load the cache content
    if file.exists():
        with file.open("r") as f:
            cache_yaml = yaml.safe_load(f)
            assoc = assoc_adapter.validate_python(cache_yaml)
            cache = {a.input: a.output for a in assoc}
    else:
        cache = {}
    # Yield the cache
    try:
        yield Cache(cache, mode)
    finally:
        # Upon destruction, write the cache back to disk
        file.parent.mkdir(parents=True, exist_ok=True)
        with file.open("w") as f:
            assoc = [Assoc(i, o) for i, o in cache.items()]
            assoc_yaml = assoc_adapter.dump_python(
                assoc, exclude_defaults=True
            )
            f.write(pretty_yaml(assoc_yaml))


@dataclass
class Assoc[P, T]:
    input: P
    output: T


type AssocList[P, T] = list[Assoc[P, T]]
