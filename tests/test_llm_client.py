"""Tests for retry behavior in app/utils/llm_client.py."""

from unittest.mock import MagicMock, patch
from google.genai import errors
from pydantic import BaseModel
import pytest
import tenacity

from app.utils.llm_client import generate_structured, is_429_error


class DummyResponseModel(BaseModel):
    """Simple Pydantic model for testing structured output parsing."""

    result: str


@pytest.fixture(autouse=True)
def zero_retry_wait(monkeypatch):
    """Eliminate backoff sleep time during unit tests to execute retries immediately."""
    monkeypatch.setattr(generate_structured.retry, "wait", tenacity.wait_none())


def test_retry_on_429_max_5_attempts():
    """Verify that a 429 ClientError is retried up to 5 attempts, then raises the final exception."""
    mock_client = MagicMock()
    mock_client.models.generate_content.side_effect = errors.ClientError(
        429, {"error": {"message": "Resource has been exhausted (e.g. check quota)."}}
    )

    with patch("app.utils.llm_client._get_client", return_value=mock_client):
        with pytest.raises(errors.ClientError) as exc_info:
            generate_structured(
                prompt="test prompt",
                system="test system",
                response_model=DummyResponseModel,
            )

    # 1. Verify final exception is the 429 ClientError
    assert exc_info.value.code == 429

    # 2. Verify exactly 5 attempts were made
    assert mock_client.models.generate_content.call_count == 5


def test_non_429_client_error_not_retried():
    """Verify that non-429 client errors (e.g., 400 Bad Request, 401 Unauthorized) are NOT retried."""
    for error_code in [400, 401, 403, 404]:
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = errors.ClientError(
            error_code, {"error": {"message": f"Client error {error_code}"}}
        )

        with patch("app.utils.llm_client._get_client", return_value=mock_client):
            with pytest.raises(errors.ClientError) as exc_info:
                generate_structured(
                    prompt="test prompt",
                    system="test system",
                    response_model=DummyResponseModel,
                )

        assert exc_info.value.code == error_code
        # Non-429 error must fail immediately on attempt 1 without retries
        assert (
            mock_client.models.generate_content.call_count == 1
        ), f"Expected 1 call for code {error_code}, got {mock_client.models.generate_content.call_count}"


def test_retry_predicate_unit():
    """Unit test for is_429_error predicate isolating 429 from other errors."""
    err_429 = errors.ClientError(429, {"error": "rate limit"})
    err_400 = errors.ClientError(400, {"error": "bad request"})
    err_500 = errors.ServerError(500, {"error": "server error"})
    err_val = ValueError("generic error")

    assert is_429_error(err_429) is True
    assert is_429_error(err_400) is False
    assert is_429_error(err_500) is False
    assert is_429_error(err_val) is False


def test_retry_success_after_transient_429():
    """Verify that if 429 occurs transiently and then succeeds within 5 attempts, result is returned."""
    mock_client = MagicMock()
    mock_success = MagicMock()
    mock_success.text = '{"result": "success"}'

    # Fails twice with 429, then succeeds on 3rd attempt
    mock_client.models.generate_content.side_effect = [
        errors.ClientError(429, {"error": "rate limit"}),
        errors.ClientError(429, {"error": "rate limit"}),
        mock_success,
    ]

    with patch("app.utils.llm_client._get_client", return_value=mock_client):
        result = generate_structured(
            prompt="test prompt",
            system="test system",
            response_model=DummyResponseModel,
        )

    assert isinstance(result, DummyResponseModel)
    assert result.result == "success"
    assert mock_client.models.generate_content.call_count == 3
