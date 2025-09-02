"""
Utilities for memoizing function calls.
"""

import functools
from collections.abc import Callable
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


@contextmanager
def load_cache(
    file: Path, *, input_type: Any, output_type: Any, mode: CacheMode
):
    """
    Load a cache from a YAML file on disk.
    """
    assoc_type = _AssocList[input_type, output_type]
    assoc_adapter = TypeAdapter[_AssocList[Any, Any]](assoc_type)
    # Load the cache content
    if file.exists():
        with file.open("r") as f:
            cache_yaml = yaml.safe_load(f)
            assoc = assoc_adapter.validate_python(cache_yaml)
            cache = {a.input: a.output for a in assoc}
    else:
        cache = {}
    # Yield the cache
    yield Cache(cache, mode)
    # Upon destruction, write the cache back to disk
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w") as f:
        assoc = [_Assoc(i, o) for i, o in cache.items()]
        assoc_yaml = assoc_adapter.dump_python(assoc)
        f.write(pretty_yaml(assoc_yaml))


@dataclass
class _Assoc[P, T]:
    input: P
    output: T


type _AssocList[P, T] = list[_Assoc[P, T]]
