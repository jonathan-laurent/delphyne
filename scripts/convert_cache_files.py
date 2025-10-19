"""
Utilities to convert cache files when the encoding scheme changes.
"""

from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import delphyne as dp
import delphyne.utils.caching as caching
from delphyne.stdlib.models import CachedRequest
from delphyne.utils.yaml import dump_yaml, load_yaml

type CacheFile = caching.AssocList[CachedRequest, dp.LLMResponse]

type Mapper = Callable[[CacheFile], CacheFile]

type RawMapper = Callable[[str], str]

type RequestProcessor = Callable[[dp.LLMRequest], dp.LLMRequest]


def find_cache_files(dir: Path) -> Iterable[Path]:
    """
    Recursively traverse `dir` to find all files of the form
    **/cache/*.yaml and **/*.cache.yaml.
    """
    yield from dir.rglob("cache/*.yaml")
    yield from dir.rglob("*.cache.yaml")


def on_requests(f: RequestProcessor) -> Mapper:
    def map(file: CacheFile) -> CacheFile:
        return [
            caching.Assoc(
                CachedRequest(f(elt.input.request), elt.input.iter), elt.output
            )
            for elt in file
        ]

    return map


def on_parsed(f: Mapper) -> RawMapper:
    def map(raw: str) -> str:
        parsed: CacheFile = load_yaml(CacheFile, raw)
        mapped = f(parsed)
        return dump_yaml(CacheFile, mapped, exclude_defaults=True)

    return map


def rewrite(dir: Path, mappers: Sequence[RawMapper]) -> None:
    files = list(find_cache_files(dir))
    for i, f in enumerate(files):
        print(f"({i + 1}/{len(files)}) Rewriting: {f}")
        raw = f.read_text()
        for mapper in mappers:
            raw = mapper(raw)
        f.write_text(raw)


if __name__ == "__main__":
    rewrite(Path("."), [on_parsed(lambda p: p)])
