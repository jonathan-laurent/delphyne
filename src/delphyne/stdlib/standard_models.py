"""
Models from standard LLM providers
"""

import os
import typing
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

from delphyne.stdlib import models as md
from delphyne.stdlib.openai_api import OpenAICompatibleModel

type OpenAIModelName = Literal[
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o4-mini"
]

type MistralModelName = Literal["mistral-small-2503", "magistral-small-2506"]

type DeepSeekModelName = Literal["deepseek-chat", "deepseek-reasoner"]

type StandardModelName = OpenAIModelName | MistralModelName | DeepSeekModelName


@dataclass
class StandardModelInfo:
    info: md.ModelInfo
    pricing: md.ModelPricing


def _default_pricing(info: md.ModelInfo) -> md.ModelPricing:
    """
    Default pricing function based on OpenAI's pricing.

    https://openai.com/api/pricing/
    """

    prices: dict[str, tuple[float, float]] = {
        "small": (0.1, 0.4),
        "medium": (0.4, 1.6),
        "large": (2.0, 8.0),
        "reasoning_small": (0.5, 1.5),  # magistral-small
        "reasoning_medium": (1.1, 4.4),  # o4-mini
        "reasoning_large": (2.0, 8.0),  # o3
    }

    inp, out = prices[str(info)]
    inp_cached = inp / 4
    return md.ModelPricing(
        dollars_per_cached_input_token=inp_cached * md.PER_MILLION,
        dollars_per_input_token=out * md.PER_MILLION,
        dollars_per_output_token=out * md.PER_MILLION,
    )


def _default_info_and_pricing(
    model: StandardModelName | str,
) -> tuple[md.ModelInfo | None, md.ModelPricing | None]:
    match model:
        case "gpt-4.1":
            info = md.ModelInfo("chat", "large")
        case "gpt-4.1-mini":
            info = md.ModelInfo("chat", "medium")
        case "gpt-4.1-nano":
            info = md.ModelInfo("chat", "small")
        case "o3":
            info = md.ModelInfo("reasoning", "large")
        case "o4-mini":
            info = md.ModelInfo("reasoning", "medium")
        case "mistral-small-2503":
            info = md.ModelInfo("chat", "small")
        case "magistral-small-2506":
            info = md.ModelInfo("chat", "small")
        case "deepseek-chat":
            info = md.ModelInfo("chat", "large")
        case "deepseek-reasoner":
            info = md.ModelInfo("reasoning", "large")
        case _:
            info = None
    pricing = _default_pricing(info) if info else None
    return info, pricing


def openai_model(model: OpenAIModelName | str):
    info, pricing = _default_info_and_pricing(model)
    return OpenAICompatibleModel(
        {"model": model}, model_info=info, pricing=pricing
    )


def mistral_model(model: MistralModelName | str):
    api_key = os.getenv("MISTRAL_API_KEY")
    assert api_key is not None
    url = "https://api.mistral.ai/v1"
    info, pricing = _default_info_and_pricing(model)
    return OpenAICompatibleModel(
        {"model": model}, api_key, url, model_info=info, pricing=pricing
    )


def deepseek_model(model: DeepSeekModelName | str):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    assert api_key is not None
    url = "https://api.deepseek.com"
    info, pricing = _default_info_and_pricing(model)
    return OpenAICompatibleModel(
        {"model": model},
        api_key,
        url,
        no_json_schema=True,
        model_info=info,
        pricing=pricing,
    )


def _values(alias: Any) -> Sequence[str]:
    return typing.get_args(alias.__value__)


def standard_model(
    model: StandardModelName,
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
