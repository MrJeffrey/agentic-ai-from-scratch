"""Pytest fixtures for tutorial tests."""

import os
from unittest.mock import patch

import pytest


@pytest.fixture
def openai_client():
    """Create OpenAI client if API key available."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY not set - skipping API tests")

    from openai import OpenAI

    return OpenAI(api_key=api_key)


@pytest.fixture
def mock_input():
    """Context manager to mock input() with predefined responses."""

    def _mock_input(responses: list[str]):
        """Returns a context manager that mocks input() with sequential responses."""
        response_iter = iter(responses + ["quit"])

        def mock_input_fn(prompt=""):
            return next(response_iter, "quit")

        return patch("builtins.input", side_effect=mock_input_fn)

    return _mock_input
