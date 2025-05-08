import asyncio

import pytest

import delphyne as dp

_TEST_PROMPT: dp.Chat = [
    dp.ChatMessage("user", "What is the capital of France?")
]


def _test_completion():
    model = dp.openai_model("gpt-4o-mini")
    print(model.send_request(_TEST_PROMPT, 2, {}))


async def _test_stream():
    model = dp.openai_model("gpt-4o-mini")
    async for res in model.stream_request(_TEST_PROMPT, {}):
        print(res)


@pytest.mark.skip(reason="OpenAI API key required")
def test_openai():
    _test_completion()
    asyncio.run(_test_stream())
