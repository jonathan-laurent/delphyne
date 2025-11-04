"""
Utilities to query Loogle
"""

import os
import random
import time
from collections.abc import Sequence
from dataclasses import dataclass

import pydantic
import requests


@dataclass
class ServerConfig:
    base_url: str = "https://loogle.lean-lang.org/"
    timeout: int = 1
    retry_base_delay: int = 10
    max_retries: int = 3
    max_shown_hits: int | None = 8

    @staticmethod
    def from_env() -> "ServerConfig":
        conf = ServerConfig()
        if (base_url := os.environ.get("LOOGLE_URL")) is not None:
            conf.base_url = base_url
        if (timeout := os.environ.get("LOOGLE_TIMEOUT")) is not None:
            conf.timeout = int(timeout)
        if (rbd := os.environ.get("LOOGLE_RETRY_BASE_DELAY")) is not None:
            conf.retry_base_delay = int(rbd)
        if (mr := os.environ.get("LOOGLE_MAX_RETRIES")) is not None:
            conf.max_retries = int(mr)
        return conf


@dataclass
class LoogleHit:
    doc: str | None
    module: str
    name: str
    type: str


@dataclass
class LoogleHits:
    """
    Header example:

        Found 125 declarations mentioning abs and HAdd.hAdd.
        Of these, 16 match your pattern(s).
    """

    count: int
    header: str
    hits: Sequence[LoogleHit]


type LoogleResults = LoogleError | LoogleHits


@dataclass
class LoogleError:
    error: str


def query_loogle(request: str) -> LoogleResults:
    config = ServerConfig.from_env()
    url = config.base_url + "/json"
    for attempt in range(config.max_retries + 1):
        try:
            response = requests.get(
                url, params={"q": request}, timeout=config.timeout
            )
            if response.status_code == 200:
                data = response.json()
                adapter = pydantic.TypeAdapter[LoogleResults](LoogleResults)
                results = adapter.validate_python(data)
                if (
                    isinstance(results, LoogleHits)
                    and config.max_shown_hits is not None
                ):
                    results.hits = results.hits[: config.max_shown_hits]
                return results
            else:
                raise Exception(
                    f"Loogle Error {response.status_code}: {response.text}"
                )
        except requests.exceptions.Timeout:
            if attempt < config.max_retries:
                base_delay = config.retry_base_delay * (2**attempt)
                jitter = random.uniform(0.5, 1.5)
                delay = base_delay * jitter
                time.sleep(delay)
            else:
                raise Exception("Loogle request timed out after retries")
    assert False


if __name__ == "__main__":
    print(query_loogle("|_ + _|"))
    print(query_loogle("{"))
