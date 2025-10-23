"""
Tools to manipulate embeddings
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast, final, override

import numpy as np
from numpy.typing import NDArray

import delphyne.core as dp
import delphyne.stdlib.models as md
import delphyne.utils.caching as caching

QUERY_EMBEDDING_PROMPT_NAME = "embed_query"
EXAMPLE_EMBEDDING_PROMPT_NAME = "embed_example"


#####
##### Embeddings Caches
#####


@dataclass(frozen=True)
class EmbeddingRequest:
    """
    A text embedding request

    Attributes:
        model: The embedding model to use.
        text: The text to embed.
    """

    model: str
    text: str


@dataclass(frozen=True)
class EmbeddingResponse:
    """
    Response to an embedding request

    Attributes:
        model: The embedding model used.
        embedding: The resulting embedding, as a one-dimensional array.
        total_tokens: The total number of tokens used to compute
    """

    model: str
    embedding: NDArray[np.float32]
    total_tokens: int


@dataclass(frozen=True)
class EmbeddingsCache(caching.Cache[EmbeddingRequest, EmbeddingResponse]):
    """
    An embedding cache, backed by an H5 file.
    """

    def save(self, file: Path) -> None:
        """
        Save all entries to the given file.

        We use the following H5 format: For each model name M, we have
        the following datasets, each with equal first dimension:
          - "$M/text": np-array of strings for `EmbeddingRequest.text`
          - "$M/model": np-array of strings for `EmbeddingResponse.model`
          - "$M/embedding": 2D np-array for `EmbeddingResponse.embedding`
          - "$M/total_tokens": np-array of ints
        """
        import h5py  # type: ignore

        # Group entries by model name
        model_groups: dict[
            str, list[tuple[EmbeddingRequest, EmbeddingResponse]]
        ] = {}
        for req, resp in self.dict.items():
            model = req.model
            if model not in model_groups:
                model_groups[model] = []
            model_groups[model].append((req, resp))

        # Save to H5 file
        with h5py.File(file, "w") as f:
            for model, entries in model_groups.items():
                if not entries:
                    continue
                # Extract data for this model
                texts = [req.text for req, _ in entries]
                models = [resp.model for _, resp in entries]
                embeddings = [resp.embedding for _, resp in entries]
                total_tokens = [resp.total_tokens for _, resp in entries]
                # Create group for this model
                f = cast(Any, f)
                group = f.create_group(model)
                # Create datasets
                # Text as variable-length strings
                dt = h5py.string_dtype(encoding="utf-8")  # type: ignore
                group.create_dataset("text", data=texts, dtype=dt)
                group.create_dataset("model", data=models, dtype=dt)
                # Embeddings as 2D array - stack all embeddings for this model
                embeddings_array = np.stack(embeddings, axis=0)
                group.create_dataset("embedding", data=embeddings_array)
                # Total tokens as integers
                group.create_dataset(
                    "total_tokens", data=total_tokens, dtype=np.int64
                )

    @staticmethod
    def load(file: Path, mode: caching.CacheMode) -> "EmbeddingsCache":
        """
        Load the cache from the given file.
        """
        import h5py  # type: ignore

        entries: dict[EmbeddingRequest, EmbeddingResponse] = {}

        with h5py.File(file, "r") as f:
            f = cast(Any, f)
            for model_name in f.keys():
                group = f[model_name]
                texts = group["text"][:]
                models = group["model"][:]
                embeddings = group["embedding"][:]
                total_tokens = group["total_tokens"][:]
                texts = [
                    t.decode("utf-8") if isinstance(t, bytes) else str(t)
                    for t in texts
                ]
                models = [
                    m.decode("utf-8") if isinstance(m, bytes) else str(m)
                    for m in models
                ]
                for i in range(len(texts)):
                    request = EmbeddingRequest(model=model_name, text=texts[i])
                    response = EmbeddingResponse(
                        model=models[i],
                        embedding=embeddings[i].astype(np.float32),
                        total_tokens=int(total_tokens[i]),
                    )
                    entries[request] = response

        return EmbeddingsCache(entries, mode)


@contextmanager
def load_embeddings_cache(file: Path, mode: caching.CacheMode):
    """
    Load an embeddings cache from a file.

    Usage:

    ```python
    with load_embeddings_cache(path, mode) as cache:
        ...
    ```
    """

    if file.exists():
        cache = EmbeddingsCache.load(file, mode)
    else:
        cache = EmbeddingsCache({}, mode)
    try:
        yield cache
    finally:
        if cache.dict or file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            cache.save(file)


#####
##### Base Models and Caching
#####


@dataclass
class EmbeddingModel(ABC):
    """
    Base class for embedding models.
    """

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @abstractmethod
    def get_max_batch_size(self) -> int:
        pass

    @abstractmethod
    def _embed(self, batch: Sequence[str]) -> Sequence[EmbeddingResponse]:
        pass

    @abstractmethod
    def spent(self, resp: EmbeddingResponse) -> dp.Budget:
        pass

    @final
    def _split_and_embed(
        self, batch: Sequence[str]
    ) -> Sequence[EmbeddingResponse]:
        if not batch:
            return []
        m = self.get_max_batch_size()
        if len(batch) <= m:
            return self._embed(batch)
        # Split the batch into chunks of max_batch_size
        results: list[EmbeddingResponse] = []
        for i in range(0, len(batch), m):
            chunk = batch[i : i + m]
            assert chunk
            chunk_results = self._embed(chunk)
            results.extend(chunk_results)
        return results

    @final
    def embed(
        self, batch: Sequence[str], cache: EmbeddingsCache | None
    ) -> Sequence[EmbeddingResponse]:
        """
        Compute a batch of embeddings, using an LLM request cache.

        This function encodes and decodes embedding requests as
        `CachedRequest` values so that the `LLMCache` infrastructure can
        be reused.
        """
        if cache is None:
            return self._split_and_embed(batch)
        model = self.get_model_name()
        reqs = [EmbeddingRequest(model, text) for text in batch]
        cached_f = cache.batched(lambda rs: self._embed([r.text for r in rs]))
        return cached_f(reqs)


#####
##### OpenAI-Compatible API
#####


# API Documentation:
# https://platform.openai.com/docs/api-reference/embeddings/create


@dataclass
class OpenAICompatibleEmbeddingModel(EmbeddingModel):
    model_name: str
    max_batch_size: int = 2048
    api_key: str | None = None
    base_url: str | None = None
    dollars_per_token: float | None = None

    @override
    def get_model_name(self) -> str:
        return self.model_name

    @override
    def get_max_batch_size(self) -> int:
        return self.max_batch_size

    @override
    def spent(self, resp: EmbeddingResponse) -> dp.Budget:
        # TODO: handle model classes to register other metrics?
        budget: dict[str, float] = {}
        if self.dollars_per_token is not None:
            budget[md.DOLLAR_PRICE] = (
                self.dollars_per_token * resp.total_tokens
            )
        return dp.Budget(budget)

    @override
    def _embed(self, batch: Sequence[str]) -> Sequence[EmbeddingResponse]:
        import openai

        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        response = client.embeddings.create(
            model=self.model_name, input=list(batch)
        )
        # There is no guarantee that embeddings are returned in order,
        # although they usually are.
        embeddings: dict[int, Sequence[float]] = {}
        for data in response.data:
            embeddings[data.index] = data.embedding
        # The Openai API only returns global token usage, while we need
        # a figure for each input separately. Thus, we build our own
        # estimate using tiktoken.
        sizes = [_count_tokens(s) for s in batch]
        return [
            EmbeddingResponse(
                model=response.model,
                embedding=np.array(embeddings[i], dtype=np.float32),
                total_tokens=sizes[i],
            )
            for i in range(len(batch))
        ]


def _count_tokens(s: str) -> int:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(s)
    return len(tokens)


#####
##### Standard Models
#####


type StandardOpenAIEmbeddingModel = Literal[
    "text-embedding-3-small",
    "text-embedding-3-large",
]


def standard_openai_embedding_model(
    name: StandardOpenAIEmbeddingModel | str,
) -> EmbeddingModel:
    match name:
        case "text-embedding-3-small":
            price = 0.02 * md.PER_MILLION
        case "text-embedding-3-large":
            price = 0.13 * md.PER_MILLION
        case _:
            raise ValueError(f"Unknown standard embedding model: {name}")
    return OpenAICompatibleEmbeddingModel(
        model_name=name, dollars_per_token=price
    )
