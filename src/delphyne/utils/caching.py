"""
Simple utility for caching values.
"""

import functools
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import get_type_hints

import yaml
from pydantic import TypeAdapter

from delphyne.utils.yaml import pretty_yaml


@dataclass
class _BucketItem[P, T]:
    input: P
    output: T


type _Bucket[P, T] = list[_BucketItem[P, T]]


def cache[P, T](
    dir: Path,
    hash_arg: Callable[[P], bytes] | None = None,
    hash_len: int = 8,
    # Pyright has trouble with more precise typing of `hash_by`
) -> Callable[[Callable[[P], T]], Callable[[P], T]]:
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
        bucket_adapter = TypeAdapter[_Bucket[P, T]](
            list[_BucketItem[arg_type, ret_type]]
        )

        @functools.wraps(func)
        def cached_func(arg: P) -> T:
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
            for b in bucket:
                if b.input == arg:
                    return b.output
            # if not found
            ret = func(arg)
            bucket.append(_BucketItem(arg, ret))
            with cache_file.open("w") as f:
                # using pretty_yaml is important because `yaml.dump`
                # uses tags to serialize tuples instead of just using
                # lists.
                f.write(pretty_yaml(bucket_adapter.dump_python(bucket)))
            return ret

        return cached_func

    return decorator
