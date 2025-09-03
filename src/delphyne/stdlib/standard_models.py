"""
Models from standard LLM providers
"""

import os
import typing
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from delphyne.stdlib import models as md
from delphyne.stdlib.openai_api import OpenAICompatibleModel

#####
##### Data about standard models
#####

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

type GeminiModelName = Literal[
    "gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"
]

type StandardModelName = (
    OpenAIModelName | MistralModelName | DeepSeekModelName | GeminiModelName
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
    "magistral-small-2506": (0.5, 0.5, 1.5),
    "deepseek-chat": (0.27, 0.07, 1.10),
    "deepseek-reasoner": (0.55, 0.14, 2.19),
    # Costs are higher above 200k tokens for Gemini.
    # We are assuming here that we stay below that threshold.
    # https://ai.google.dev/gemini-api/docs/pricing
    "gemini-2.5-pro": (1.25, 0.31, 10.00),
    "gemini-2.5-flash": (0.30, 0.075, 2.50),
    "gemini-2.5-flash-lite": (0.10, 0.025, 0.40),
}


def test_pricing_dict_exhaustiveness():
    # To be called within the test suite.
    pricing_keys = set(PRICING.keys())
    literal_values = set(
        [
            *_values(OpenAIModelName),
            *_values(MistralModelName),
            *_values(DeepSeekModelName),
            *_values(GeminiModelName),
        ]
    )
    not_in_pricing = literal_values - pricing_keys
    assert not_in_pricing == set(), (
        f"Models are missing from standard_models.PRICING: {not_in_pricing}"
    )
    not_in_literals = pricing_keys - literal_values
    assert not_in_literals == set(), (
        f"Extra models found in standard_models.PRICING: {not_in_literals}"
    )


def _values(alias: Any) -> Sequence[str]:
    """
    Return the possible values of a literal type alias.
    """
    return typing.get_args(alias.__value__)


def _standard_model_names() -> Iterable[str]:
    return PRICING.keys()


def _longest_standard_model_prefix_or_self(model_name: str) -> str:
    all = _standard_model_names()
    cands = [m for m in all if model_name.startswith(m)]
    if not cands:
        return model_name
    return max(cands, key=len)  # Return the longest matching candidate


#####
##### Utilities for building standard models
#####


def _get_pricing(model_name: str) -> md.ModelPricing | None:
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
    return None


def _openai_compatible_model(
    model: str,
    *,
    options: md.RequestOptions | None = None,
    pricing: md.ModelPricing | None | Literal["auto"] = "auto",
    model_class: str | None = None,
    base_url: str,
    api_key_env_var: str,
):
    """
    Build a model accessible from an OpenAI-compatible API. See
    `standard_model` for details on all parameters.

    Parameters:
        base_url: the base URL for the API, e.g.,
            "https://api.openai.com/v1" for OpenAI.
        api_key_env_var: the name of the environment variable
            containing the API key, e.g., "OPENAI_API_KEY".
    """

    api_key = os.getenv(api_key_env_var)
    assert api_key is not None, (
        f"Please set environment variable {api_key_env_var}."
    )
    if pricing == "auto":
        pricing = _get_pricing(_longest_standard_model_prefix_or_self(model))
        if pricing is None:
            raise ValueError(
                f"Pricing information could not be inferred for {model}."
            )
    all_options: md.RequestOptions = {"model": model}
    if options is not None:
        all_options.update(options)
    return OpenAICompatibleModel(
        base_url=base_url,
        api_key=api_key,
        options=all_options,
        model_class=model_class,
        pricing=pricing,
    )


def openai_model(
    model: OpenAIModelName | str,
    *,
    options: md.RequestOptions | None = None,
    pricing: md.ModelPricing | None | Literal["auto"] = "auto",
    model_class: str | None = None,
):
    """
    Obtain a standard model from OpenAI.

    See `standard_model` for details.
    """
    return _openai_compatible_model(
        model,
        options=options,
        pricing=pricing,
        model_class=model_class,
        base_url="https://api.openai.com/v1",
        api_key_env_var="OPENAI_API_KEY",
    )


def mistral_model(
    model: MistralModelName | str,
    *,
    options: md.RequestOptions | None = None,
    pricing: md.ModelPricing | None | Literal["auto"] = "auto",
    model_class: str | None = None,
):
    """
    Obtain a standard model from Mistral.

    See `standard_model` for details.
    """
    return _openai_compatible_model(
        model,
        options=options,
        pricing=pricing,
        model_class=model_class,
        base_url="https://api.mistral.ai/v1",
        api_key_env_var="MISTRAL_API_KEY",
    )


def deepseek_model(
    model: DeepSeekModelName | str,
    *,
    options: md.RequestOptions | None = None,
    pricing: md.ModelPricing | None | Literal["auto"] = "auto",
    model_class: str | None = None,
):
    """
    Obtain a standard model from DeepSeek.

    See `standard_model` for details.
    """
    return _openai_compatible_model(
        model,
        options=options,
        pricing=pricing,
        model_class=model_class,
        base_url="https://api.deepseek.com",
        api_key_env_var="DEEPSEEK_API_KEY",
    )


def gemini_model(
    model: GeminiModelName | str,
    *,
    options: md.RequestOptions | None = None,
    pricing: md.ModelPricing | None | Literal["auto"] = "auto",
    model_class: str | None = None,
):
    """
    Obtain a standard model from Gemini.

    See `standard_model` for details.
    """
    return _openai_compatible_model(
        model,
        options=options,
        pricing=pricing,
        model_class=model_class,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env_var="GEMINI_API_KEY",
    )


def standard_model(
    model: StandardModelName | str,
    *,
    options: md.RequestOptions | None = None,
    pricing: md.ModelPricing | None | Literal["auto"] = "auto",
    model_class: str | None = None,
) -> OpenAICompatibleModel:
    """
    Obtain a standard model from OpenAI, Mistral, DeepSeek or Gemini.

    Make sure that the following environment variables are set:

    - `OPENAI_API_KEY` for OpenAI models
    - `MISTRAL_API_KEY` for Mistral models
    - `DEEPSEEK_API_KEY` for DeepSeek models
    - `GEMINI_API_KEY` for Gemini models

    Parameters:
        model: The name of the model to use. The model provider is
            automatically inferred from this name.
        options: Additional options for the model, such as reasoning
            effort or default temperature. The `model` option must not
            be overriden.
        pricing: Pricing model to use. If `"auto"` is provided
            (default), it is inferred from the model's name (or
            `ValueError` is raised). If `None` is provided, no pricing
            information is used and so the associated budget metrics
            won't be computed.
        model_class: An optional identifier for the model class (e.g.,
            "reasoning_large"). When provided, class-specific budget
            metrics are reported, so that resource consumption can be
            tracked separately for different classes of models (e.g.,
            tracking "num_requests__reasoning_large" separately from
            "num_requests__chat_small").

    Raises:
        ValueError: The provider or pricing model could not be inferred.
    """

    openai_models = _values(OpenAIModelName)
    mistral_models = _values(MistralModelName)
    deepseek_models = _values(DeepSeekModelName)
    gemini_models = _values(GeminiModelName)

    prefix = _longest_standard_model_prefix_or_self(model)

    if prefix in openai_models:
        make_model = openai_model
    elif prefix in mistral_models:
        make_model = mistral_model
    elif prefix in deepseek_models:
        make_model = deepseek_model
    elif prefix in gemini_models:
        make_model = gemini_model
    else:
        raise ValueError(
            f"Failed to infer provider for model: {model}.\n"
            + "Use a more specific function such as `openai_model`."
        )
    return make_model(
        model, options=options, pricing=pricing, model_class=model_class
    )
