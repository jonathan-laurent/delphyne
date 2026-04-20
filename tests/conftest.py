from typing import Any
from unittest.mock import patch

import pytest

import delphyne as dp


@pytest.fixture(params=["chat_completions", "responses"], autouse=True)
def _parameterize_api_type(request: pytest.FixtureRequest):  # pyright: ignore[reportUnusedFunction]
    """
    Parameterizes the API type to be used for tests.

    This fixture is used to inject `api_type` parameterization into
    `standard_model` when `api_type` is not already specified as an argument.
    It also adds an `openai_responses_model` version for every
    test using `openai_model`. It is applied to all tests in the test suite.
    """
    api_type: str = request.param
    original_standard_model = dp.standard_model
    original_openai_model = dp.openai_model

    def wrapped_standard_model(model: str, *args: Any, **kwargs: Any):
        if "api_type" not in kwargs:
            kwargs["api_type"] = api_type

        try:
            return original_standard_model(model, *args, **kwargs)
        except ValueError as e:
            if (
                api_type == "responses"
                and "Responses API is only supported" in str(e)
            ):
                pytest.skip(f"Skipping: {e}")
            raise

    def wrapped_openai_model(model: str, *args: Any, **kwargs: Any):
        if api_type == "responses":
            return dp.openai_responses_model(model, *args, **kwargs)
        else:
            return original_openai_model(model, *args, **kwargs)

    with (
        patch("delphyne.standard_model", side_effect=wrapped_standard_model),
        patch("delphyne.openai_model", side_effect=wrapped_openai_model),
        patch(
            "delphyne.stdlib.standard_models.standard_model",
            side_effect=wrapped_standard_model,
        ),
        patch(
            "delphyne.stdlib.standard_models.openai_model",
            side_effect=wrapped_openai_model,
        ),
    ):
        yield api_type
