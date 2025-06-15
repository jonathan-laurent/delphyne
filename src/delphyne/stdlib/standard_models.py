"""
Models from standard LLM providers
"""

import os
import typing
from collections.abc import Sequence
from typing import Any, Literal

from delphyne.stdlib.openai_api import OpenAICompatibleModel

type OpenAIModelName = Literal[
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o4-mini"
]

type MistralModelName = Literal["mistral-small-2503", "magistral-small-2506"]

type DeepSeekModelName = Literal["deepseek-chat", "deepseek-reasoner"]

type StandardModelName = OpenAIModelName | MistralModelName | DeepSeekModelName


def openai_model(model: OpenAIModelName | str):
    return OpenAICompatibleModel({"model": model})


def mistral_model(model: MistralModelName | str):
    api_key = os.getenv("MISTRAL_API_KEY")
    assert api_key is not None
    url = "https://api.mistral.ai/v1"
    return OpenAICompatibleModel({"model": model}, api_key, url)


def deepseek_model(model: DeepSeekModelName | str):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    assert api_key is not None
    url = "https://api.deepseek.com"
    return OpenAICompatibleModel({"model": model}, api_key, url)


def _values(alias: Any) -> Sequence[str]:
    return typing.get_args(alias.__value__)


def standard_model(
    model: OpenAIModelName | MistralModelName | DeepSeekModelName,
) -> OpenAICompatibleModel:
    openai_models = _values(OpenAIModelName)
    mistral_models = _values(MistralModelName)
    deepseek_models = _values(DeepSeekModelName)
    if model in openai_models:
        return openai_model(model)
    elif model in mistral_models:
        return mistral_model(model)
    elif model in deepseek_models:
        return deepseek_model(model)
    assert False, f"Unknown model: {model}"
