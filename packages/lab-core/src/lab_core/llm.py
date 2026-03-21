"""LLM client utilities."""

from __future__ import annotations

import os

from openai import OpenAI


def get_vllm_client(base_url: str = "http://localhost:8000/v1") -> OpenAI:
    """Get an OpenAI client pointing at the local vLLM instance."""
    return OpenAI(base_url=base_url, api_key="not-needed")


def get_gateway_client(
    base_url: str = "http://localhost:4000/v1",
    api_key: str | None = None,
) -> OpenAI:
    """Get an OpenAI client pointing at the LiteLLM gateway."""
    key = api_key or os.environ.get("GATEWAY_API_KEY", "not-needed")
    return OpenAI(base_url=base_url, api_key=key)
