"""
Models from standard LLM providers
"""

import os
import typing
from collections.abc import Sequence
from typing import Literal, cast

from delphyne.stdlib.openai_api import OpenAICompatibleModel

type OpenAIModelName = Literal[
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o4-mini"
]

type MistralModelName = Literal["mistral-small-2503", "magistral-small-2506"]

type DeepSeekModelName = Literal["deepseek-chat", "deepseek-reasoner"]


def openai_model(model: OpenAIModelName | str):
    return OpenAICompatibleModel({"model": model})


def mistral_model(model: MistralModelName | str):
    api_key = os.getenv("MISTRAL_API_KEY")
    assert api_key is not None
    url = "https://api.mistral.ai/v1/chat/completions"
    return OpenAICompatibleModel({"model": model}, api_key, url)


def deepseek_model(model: DeepSeekModelName | str):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    assert api_key is not None
    url = "https://api.deepseek.com"
    return OpenAICompatibleModel({"model": model}, api_key, url)


def standard_model(
    model: OpenAIModelName | MistralModelName | DeepSeekModelName,
):
    openai_models = cast(Sequence[str], typing.get_args(OpenAIModelName))
    mistral_models = cast(Sequence[str], typing.get_args(MistralModelName))
    deepseek_models = cast(Sequence[str], typing.get_args(DeepSeekModelName))
    if model in openai_models:
        return openai_model(model)
    elif model in mistral_models:
        return mistral_model(model)
    elif model in deepseek_models:
        return deepseek_model(model)
    assert False, f"Unknown model: {model}"
