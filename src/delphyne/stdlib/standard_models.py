"""
Models from standard LLM providers
"""

from typing import Literal

from delphyne.stdlib.openai_api import OpenAICompatibleModel

type OpenAIModelName = Literal[
    "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano", "o3", "o4-mini"
]


def openai_model(model: OpenAIModelName | str):
    return OpenAICompatibleModel({"model": model})
