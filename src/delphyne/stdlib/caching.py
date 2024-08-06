"""
Simple utility for caching values.
"""

import functools
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import get_type_hints

import yaml
from pydantic import TypeAdapter

from delphyne.utils.yaml import pretty_yaml


HASH_LEN = 8


def _short_hash(s: bytes) -> str:
    return hashlib.md5(s).hexdigest()[:HASH_LEN]


type Bucket[P, T] = list[tuple[P, T]]


def cache[P, T](dir: Path) -> Callable[[Callable[[P], T]], Callable[[P], T]]:
    """
    A decorator that adds hard-disk caching to a function.

    When presented with an argument, the cached function converts this
    argument to YAML and computes a hash of the resulting YAML string.
    The `dir` directory contains a memoization hash-table, where each
    bucket is a file named after the corresponding hash. We use pydantic
    for serialization/deserializationm, `using pydantic.TypeAdapter` and
    `typing.get_type_hints` to automatically derive the type of P.
    """

    def decorator(func: Callable[[P], T]) -> Callable[[P], T]:

        arg_types = list(get_type_hints(func).items())
        assert len(arg_types) == 2, "Function must have exactly one argument"
        arg_type = arg_types[0][1]
        ret_type = arg_types[1][1]
        arg_adapter = TypeAdapter[P](arg_type)
        bucket_adapter = TypeAdapter[Bucket[P, T]](
            list[tuple[arg_type, ret_type]]
        )

        @functools.wraps(func)
        def cached_func(arg: P) -> T:
            dir.mkdir(parents=True, exist_ok=True)
            arg_json = arg_adapter.dump_json(arg)
            arg_hash = _short_hash(arg_json)
            cache_file = dir / f"{arg_hash}.yaml"
            if cache_file.exists():
                with cache_file.open("r") as f:
                    bucket_yaml = yaml.safe_load(f)
                    bucket = bucket_adapter.validate_python(bucket_yaml)
            else:
                bucket: Bucket[P, T] = []
            for arg_, ret in bucket:
                if arg_ == arg:
                    return ret
            # if not found
            ret = func(arg)
            bucket.append((arg, ret))
            with cache_file.open("w") as f:
                # using pretty_yaml is important because `yaml.dump`
                # uses tags to serialize tuples instead of just using
                # lists.
                f.write(pretty_yaml(bucket_adapter.dump_python(bucket)))
            return ret

        return cached_func

    return decorator
