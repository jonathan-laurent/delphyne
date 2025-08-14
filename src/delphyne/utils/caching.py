"""
Utilities for memoizing function calls on disk.
"""

import functools
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_type_hints

import yaml
from pydantic import TypeAdapter

from delphyne.utils.pretty_yaml import pretty_yaml

type CacheMode = Literal["read_write", "off", "create", "replay"]
"""
Caching mode:

- `off`: the cache is disabled
- `read_write`: values can be read and written to the cache (no extra
      check is made)
- `create`: the cache is used in write-only mode, and an exception is
      raided if a cached value already exists.
- `replay`: all requests must hit the cache or an exception is raised.
"""


def cache_database_file(dir: Path) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    return dir / DATABASE_FILE_NAME


#####
##### DBM Caching
#####


DATABASE_FILE_NAME = "cache"  # dbm will add the *.db extension


def cache_db[P, T](
    database: Any,
    mode: CacheMode = "read_write",
) -> Callable[[Callable[[P], T]], Callable[[P], T]]:
    def decorator(func: Callable[[P], T]) -> Callable[[P], T]:
        arg_type, ret_type = _inspect_arg_and_ret_types(func)
        arg_adapter = TypeAdapter[P](arg_type)
        ret_adapter = TypeAdapter[T](ret_type)

        @functools.wraps(func)
        def cached_func(arg: P) -> T:
            if mode == "off":
                return func(arg)
            arg_json = arg_adapter.dump_json(arg)
            if arg_json in database:
                assert mode != "create", "Cache entry already exists."
                ret_json = database[arg_json]
                ret = ret_adapter.validate_json(ret_json)
                return ret
            assert mode != "replay", f"Cache entry not found for:\n\n {arg}"
            ret = func(arg)
            ret_json = ret_adapter.dump_json(ret)
            database[arg_json] = ret_json
            return ret

        return cached_func

    return decorator


def _inspect_arg_and_ret_types(func: Callable[[Any], Any]) -> tuple[Any, Any]:
    arg_types = list(get_type_hints(func).items())
    assert len(arg_types) == 2, "Function must have exactly one argument"
    arg_type = arg_types[0][1]
    ret_type = arg_types[1][1]
    return arg_type, ret_type


#####
##### YAML Caching
#####


@dataclass
class _BucketItem[P, T]:
    input: P
    output: T


type _Bucket[P, T] = list[_BucketItem[P, T]]


def cache_yaml[P, T](
    dir: Path,
    hash_arg: Callable[[P], bytes] | None = None,
    hash_len: int = 8,
    mode: CacheMode = "read_write",
) -> Callable[[Callable[[P], T]], Callable[[P], T]]:
    """
    A decorator that adds hard-disk caching to a function.

    When presented with an argument, the cached function converts this
    argument to YAML and computes a hash of the resulting YAML string.
    The `dir` directory contains a memoization hash-table, where each
    bucket is a file named after the corresponding hash. We use pydantic
    for serialization/deserialization, `using pydantic.TypeAdapter` and
    `typing.get_type_hints` to automatically derive the type of P.

    Note that decorating a function is fast and does not perform any
    operation on disk (these are only performed when executing the
    cached function).
    """

    def decorator(func: Callable[[P], T]) -> Callable[[P], T]:
        arg_type, ret_type = _inspect_arg_and_ret_types(func)
        arg_adapter = TypeAdapter[P](arg_type)
        bucket_adapter = TypeAdapter[_Bucket[P, T]](
            list[_BucketItem[arg_type, ret_type]]
        )

        @functools.wraps(func)
        def cached_func(arg: P) -> T:
            if mode == "off":
                return func(arg)
            dir.mkdir(parents=True, exist_ok=True)
            if hash_arg is not None:
                arg_bytes = hash_arg(arg)
            else:
                arg_bytes = arg_adapter.dump_json(arg)
            arg_hash = hashlib.md5(arg_bytes).hexdigest()
            arg_hash = arg_hash[:hash_len]
            cache_file = dir / f"{arg_hash}.yaml"
            if cache_file.exists():
                with cache_file.open("r") as f:
                    bucket_yaml = yaml.safe_load(f)
                    bucket = bucket_adapter.validate_python(bucket_yaml)
            else:
                bucket: _Bucket[P, T] = []
            # if `arg.x: Sequence[int] = ()`, then parsing `arg` again
            # might make it an empty list instead so we test equality
            # with the roundabout conversion.
            arg_roundabout = arg_adapter.validate_json(
                arg_adapter.dump_json(arg)
            )
            for b in bucket:
                if b.input == arg_roundabout:
                    assert mode != "create", "Cache entry already exists."
                    return b.output
            assert mode != "replay", f"Cache entry not found for:\n\n {arg}"
            # if not found, we execute the function
            ret = func(arg)
            bucket.append(_BucketItem(arg, ret))
            with cache_file.open("w") as f:
                bucket_py = bucket_adapter.dump_python(bucket)
                f.write(pretty_yaml(bucket_py))
            return ret

        return cached_func

    return decorator


#####
##### Generic Interface
#####


@dataclass(frozen=True)
class CacheDb:
    """
    Wraps a dbm cache database. Build using:

        with open(caching.cache_database_file(dir), "c") as db:
            info = CacheDb(db)
            ...
    """

    database: Any  # dbm._Database


@dataclass(frozen=True)
class CacheYaml:
    """
    Use a human-readable, YAML-based database on disk.

    Attributes:
        cache_dir: Path to the directory where the YAML database files
            are stored.
    """

    cache_dir: Path


type CacheInfo = CacheDb | CacheYaml


@dataclass(frozen=True)
class CacheSpec:
    """
    Specification for a function cache.

    Attributes:
        info: Whether to use a YAML or DBM database, and where this
            database is located.
        mode: Desired caching behavior.
    """

    info: CacheInfo
    mode: CacheMode = "read_write"


def cache[P, T](
    cache_spec: CacheSpec,
    hash_arg: Callable[[P], bytes] | None = None,
    hash_len: int = 8,
) -> Callable[[Callable[[P], T]], Callable[[P], T]]:
    match cache_spec.info:
        case CacheDb(database):
            return cache_db(database, cache_spec.mode)
        case CacheYaml(cache_dir):
            return cache_yaml(
                cache_dir,
                mode=cache_spec.mode,
                hash_arg=hash_arg,
                hash_len=hash_len,
            )
