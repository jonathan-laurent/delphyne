"""
Models from standard LLM providers
"""

import os
import typing
from collections.abc import Sequence
from typing import Any, Literal, TypeGuard

from delphyne.core.inspect import literal_type_args
from delphyne.stdlib import models as md
from delphyne.stdlib.openai_api import OpenAICompatibleModel

type OpenAIModelName = Literal[
    "gpt-5",
    "gpt-5-mini",
    "gpt-5-nano",
    "gpt-4.1",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4.1-mini",
    "gpt-4.1-nano",
    "o3",
    "o4-mini",
]

type MistralModelName = Literal["mistral-small-2503", "magistral-small-2506"]

type DeepSeekModelName = Literal["deepseek-chat", "deepseek-reasoner"]

type StandardModelName = OpenAIModelName | MistralModelName | DeepSeekModelName


def is_standard_model_name(
    model_name: str,
) -> TypeGuard[StandardModelName]:
    """
    Check if a string is a standard model name.
    """
    openai_models = literal_type_args(OpenAIModelName)
    mistral_models = literal_type_args(MistralModelName)
    deepseek_models = literal_type_args(DeepSeekModelName)

    return (
        (openai_models is not None and model_name in openai_models)
        or (mistral_models is not None and model_name in mistral_models)
        or (deepseek_models is not None and model_name in deepseek_models)
    )


PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-5": (1.25, 0.125, 10.00),  # cached input 10x less expensive!
    "gpt-5-mini": (0.250, 0.025, 2.00),
    "gpt-5-nano": (0.050, 0.005, 0.40),
    "gpt-4.1": (1.10, 0.275, 4.40),
    "gpt-4.1-mini": (0.80, 0.20, 3.20),
    "gpt-4.1-nano": (0.20, 0.05, 0.80),
    "gpt-4o": (2.50, 1.25, 10.00),
    "gpt-4o-mini": (0.15, 0.075, 0.60),  # cached input = input ×50% ⇒ 0.075
    "o4-mini": (4.00, 1.00, 16.00),
    "o3": (10.00, 2.50, 40.00),
    "mistral-small-2503": (0.10, 0.10, 0.30),
    "deepseek-chat": (0.27, 0.07, 1.10),
    "deepseek-reasoner": (0.55, 0.14, 2.19),
}


def get_pricing(model_name: str) -> md.ModelPricing:
    """
    Get the pricing for a model by its name.
    Returns None if the model is not found.
    """
    if model_name in PRICING:
        inp, out, out_cached = PRICING[model_name]
        return md.ModelPricing(
            dollars_per_cached_input_token=out_cached * md.PER_MILLION,
            dollars_per_input_token=inp * md.PER_MILLION,
            dollars_per_output_token=out * md.PER_MILLION,
        )
    assert False, f"Unknown model: {model_name}"


def openai_model(
    model: OpenAIModelName | str,
    *,
    options: md.RequestOptions | None = None,
    pricing: md.ModelPricing | None = None,
    model_class: str | None = None,
):
    if pricing is None:
        pricing = get_pricing(model)
    all_options: md.RequestOptions = {"model": model}
    if options is not None:
        all_options.update(options)
    return OpenAICompatibleModel(
        options=all_options, model_class=model_class, pricing=pricing
    )


def mistral_model(
    model: MistralModelName | str,
    *,
    options: md.RequestOptions | None = None,
    pricing: md.ModelPricing | None = None,
    model_class: str | None = None,
):
    api_key = os.getenv("MISTRAL_API_KEY")
    url = "https://api.mistral.ai/v1"
    if pricing is None:
        pricing = get_pricing(model)
    all_options: md.RequestOptions = {"model": model}
    if options is not None:
        all_options.update(options)
    return OpenAICompatibleModel(
        options=all_options,
        api_key=api_key,
        base_url=url,
        pricing=pricing,
        model_class=model_class,
    )


def deepseek_model(
    model: DeepSeekModelName | str,
    *,
    options: md.RequestOptions | None = None,
    pricing: md.ModelPricing | None = None,
    model_class: str | None = None,
):
    api_key = os.getenv("DEEPSEEK_API_KEY")
    url = "https://api.deepseek.com"
    if pricing is None:
        pricing = get_pricing(model)
    all_options: md.RequestOptions = {"model": model}
    if options is not None:
        all_options.update(options)
    return OpenAICompatibleModel(
        options=all_options,
        api_key=api_key,
        base_url=url,
        no_json_schema=True,
        pricing=pricing,
        model_class=model_class,
    )


def _values(alias: Any) -> Sequence[str]:
    return typing.get_args(alias.__value__)


def standard_model(
    model: StandardModelName,
    *,
    options: md.RequestOptions | None = None,
    pricing: md.ModelPricing | None = None,
    model_class: str | None = None,
) -> OpenAICompatibleModel:
    """
    Obtain a standard model from OpenAI, Mistral or DeepSeek.

    Make sure that the following environment variables are set:

    - `OPENAI_API_KEY` for OpenAI models
    - `MISTRAL_API_KEY` for Mistral models
    - `DEEPSEEK_API_KEY` for DeepSeek models

    Attributes:
        model: The name of the model to use.
        options: Additional options for the model, such as reasoning
            effort or default temperature. The `model` option must not
            be overriden.
        pricing: Use a custom pricing model, if provided.
        model_class: an optional identifier for the model class (e.g.,
            "reasoning_large"). When provided, class-specific budget
            metrics are reported, so that resource consumption can be
            tracked separately for different classes of models (e.g.,
            tracking "num_requests__reasoning_large" separately from
            "num_requests__chat_small").
    """
    openai_models = _values(OpenAIModelName)
    mistral_models = _values(MistralModelName)
    deepseek_models = _values(DeepSeekModelName)
    if model in openai_models:
        return openai_model(
            model, options=options, pricing=pricing, model_class=model_class
        )
    elif model in mistral_models:
        return mistral_model(
            model, options=options, pricing=pricing, model_class=model_class
        )
    elif model in deepseek_models:
        return deepseek_model(
            model, options=options, pricing=pricing, model_class=model_class
        )
    else:
        assert False, f"Unknown model: {model}"
