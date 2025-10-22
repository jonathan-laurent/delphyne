"""
Tools to manipulate embeddings
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, final, override

import delphyne.core as dp
import delphyne.stdlib.models as md

EMBEDDING_PROMPT_NAME = "embed"


#####
##### Base Models and Caching
#####


@dataclass
class EmbeddingResponse:
    model: str
    embedding: Sequence[float]
    total_tokens: int


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
        self, batch: Sequence[str], cache: md.LLMCache | None
    ) -> Sequence[EmbeddingResponse]:
        """
        Compute a batch of embeddings, using an LLM request cache.

        This function encodes and decodes embedding requests as
        `CachedRequest` values so that the `LLMCache` infrastructure can
        be reused.
        """
        if cache is None:
            return self._split_and_embed(batch)

        def compute_batch(
            reqs: Sequence[md.CachedRequest],
        ) -> Sequence[md.LLMResponse]:
            batch = [_decode_embedding_request(r) for r in reqs]
            resps = self._embed(batch)
            return [_encode_embedding_response(r) for r in resps]

        reqs = _encode_embedding_requests(batch, self.get_model_name())
        resps = cache.cache.batched(compute_batch)(reqs)
        return [_decode_embedding_response(r) for r in resps]


def _encode_embedding_requests(
    batch: Sequence[str], model: str
) -> Sequence[md.CachedRequest]:
    # Embeddings are cached once and for all (we set `iter` to a constant).
    return [
        md.CachedRequest(
            md.LLMRequest(
                chat=(md.UserMessage(content=elt),),
                num_completions=1,
                options={"model": model},
            ),
            iter=0,
        )
        for elt in batch
    ]


def _decode_embedding_request(req: md.CachedRequest) -> str:
    message = req.request.chat[0]
    assert isinstance(message, md.UserMessage)
    return message.content


def _encode_embedding_response(resp: EmbeddingResponse) -> md.LLMResponse:
    output = md.LLMOutput(dp.Structured(resp.embedding))
    return md.LLMResponse(
        [output],
        model_name=resp.model,
        usage_info={"total_tokens": resp.total_tokens},
    )


def _decode_embedding_response(resp: md.LLMResponse) -> EmbeddingResponse:
    content = resp.outputs[0].content
    assert isinstance(content, dp.Structured)
    embedding = content.structured
    assert resp.model_name is not None
    assert resp.usage_info is not None
    return EmbeddingResponse(
        model=resp.model_name,
        embedding=embedding,
        total_tokens=resp.usage_info["total_tokens"],
    )


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
                embedding=embeddings[i],
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
