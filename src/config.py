"""Shared configuration and model factory."""

from __future__ import annotations

import os

from dotenv import load_dotenv
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

load_dotenv()

PROXY_URL = "https://litellm.6640.ucf.spencerlyon.com"
DEFAULT_MODEL = "claude-haiku-4-5"


def get_model(model_name: str = DEFAULT_MODEL) -> OpenAIChatModel:
    """Create an OpenAI-compatible model routed through class proxy."""
    provider = OpenAIProvider(base_url=PROXY_URL, api_key=os.environ["CAP6640_API_KEY"])
    return OpenAIChatModel(model_name=model_name, provider=provider)
