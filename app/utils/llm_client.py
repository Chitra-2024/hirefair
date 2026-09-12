"""Shared Google Gemini structured-output utility for HireFair."""

import os
from typing import Type
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_exponential

load_dotenv()


def is_429_error(exception: BaseException) -> bool:
    """Return True if exception is a google.genai.errors.ClientError with code 429."""
    return isinstance(exception, errors.ClientError) and getattr(exception, "code", None) == 429


# Alias for readability
is_rate_limit_error = is_429_error


def _get_client() -> genai.Client:
    """Return an initialized Google GenAI client reading from environment."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        return genai.Client(api_key=api_key)
    return genai.Client()


@retry(
    retry=retry_if_exception(is_429_error),
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    reraise=True,
)
def generate_structured(
    prompt: str,
    system: str,
    response_model: Type[BaseModel],
    model: str = "gemini-3.6-flash",
) -> BaseModel:
    """Generate structured output validated against a Pydantic model using Gemini.

    Args:
        prompt: User input prompt / content for the model.
        system: System instruction guiding the model behavior.
        response_model: Pydantic model class for schema constraint and validation.
        model: Model identifier (default: gemini-3.6-flash).

    Returns:
        An instance of response_model populated from the parsed JSON.
    """
    client = _get_client()

    config = types.GenerateContentConfig(
        system_instruction=system,
        response_mime_type="application/json",
        response_schema=response_model,
        temperature=0.0,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return response_model.model_validate_json(response.text)
