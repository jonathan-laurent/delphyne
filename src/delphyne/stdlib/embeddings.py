"""
Tools to manipulate embeddings
"""

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import final

import delphyne.core as dp
import delphyne.stdlib.models as md
from delphyne.utils.yaml import pretty_yaml

#####
##### Models
#####


@dataclass
class EmbeddingResponse:
    model: str
    embeddings: Sequence[Sequence[float]]
    total_tokens: int


@dataclass
class EmbeddingModel(ABC):
    """
    Base class for embedding models.
    """

    @abstractmethod
    def get_model_name(self) -> str:
        pass

    @final
    def embed_with_cache(
        self, batch: Sequence[str], cache: md.LLMCache | None
    ) -> EmbeddingResponse:
        req = _encode_embedding_request(batch, self.get_model_name())
        resp = md.fetch_or_answer(
            cache, req, lambda _: _encode_embedding_response(self.embed(batch))
        )
        return _decode_embedding_response(resp)

    @abstractmethod
    def embed(self, batch: Sequence[str]) -> EmbeddingResponse:
        pass


def _encode_embedding_request(
    batch: Sequence[str], model: str
) -> md.LLMRequest:
    content = pretty_yaml(batch)
    return md.LLMRequest(
        chat=(md.UserMessage(content=content),),
        num_completions=1,
        options={"model": model},
    )


def _encode_embedding_response(resp: EmbeddingResponse) -> md.LLMResponse:
    output = md.LLMOutput(dp.Structured(resp.embeddings))
    return md.LLMResponse(
        [output],
        model_name=resp.model,
        usage_info={"total_tokens": resp.total_tokens},
    )


def _decode_embedding_response(resp: md.LLMResponse) -> EmbeddingResponse:
    content = resp.outputs[0].content
    assert isinstance(content, dp.Structured)
    embeddings = content.structured
    assert resp.model_name is not None
    assert resp.usage_info is not None
    return EmbeddingResponse(
        model=resp.model_name,
        embeddings=embeddings,
        total_tokens=resp.usage_info["total_tokens"],
    )


#####
#####
#####

# API Documentation:
# https://platform.openai.com/docs/api-reference/embeddings/create
