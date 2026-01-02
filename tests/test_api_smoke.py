"""API smoke tests: Verify OpenAI SDK patterns work (single API call per pattern)."""

import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)


def test_chat_completions_basic(openai_client):
    """Verify basic chat completions API works (Tutorial 1 pattern)."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a test assistant."},
            {"role": "user", "content": "Say 'test passed' and nothing else."},
        ],
        max_tokens=10,
    )
    assert response.choices[0].message.content is not None
    assert response.usage.total_tokens > 0


def test_tool_calling_basic(openai_client):
    """Verify tool calling API works (Tutorial 2 pattern)."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "test_tool",
                "description": "A test tool",
                "parameters": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "required": ["value"],
                },
            },
        }
    ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Call test_tool with value 'hello'"}],
        tools=tools,
        tool_choice="required",
        max_tokens=50,
    )

    assert response.choices[0].message.tool_calls is not None
    assert len(response.choices[0].message.tool_calls) > 0
    tool_call = response.choices[0].message.tool_calls[0]
    assert tool_call.function.name == "test_tool"
    assert tool_call.function.arguments is not None


def test_embeddings_basic(openai_client):
    """Verify embeddings API works (Tutorial 5 pattern)."""
    response = openai_client.embeddings.create(
        model="text-embedding-3-small",
        input="test embedding",
    )

    assert len(response.data) == 1
    assert len(response.data[0].embedding) > 0


def test_json_response_format(openai_client):
    """Verify JSON response format works (Tutorial 8 pattern)."""
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "Return JSON with key 'status' set to 'ok'.",
            },
            {"role": "user", "content": "Give me the status."},
        ],
        response_format={"type": "json_object"},
        max_tokens=20,
    )

    content = response.choices[0].message.content
    data = json.loads(content)
    assert "status" in data


def test_openai_sdk_version():
    """Verify OpenAI SDK version meets minimum requirement."""
    import openai
    from packaging import version

    min_version = "1.0.0"
    current = openai.__version__
    assert version.parse(current) >= version.parse(min_version), (
        f"OpenAI SDK {current} is below minimum {min_version}"
    )
