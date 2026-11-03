from typing import override
from unittest.mock import MagicMock, patch

import httpx
import pytest
from litellm.exceptions import RateLimitError

from openhands.resolver.interfaces.github import GithubIssueHandler, GithubPRHandler
from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.interfaces.issue_definitions import ServiceContextIssue, ServiceContextPR
from tests.unit.resolver.compat import LLM, LLMConfig, MessageAction


@pytest.fixture
def default_config() -> object:
    return LLMConfig(
        model="gpt-4o",
        api_key="test_key",
        num_retries=2,
        retry_min_wait=1,
        retry_max_wait=2,
    )


def test_handle_nonexistent_issue_reference() -> None:
    llm_config = LLMConfig(model="test", api_key="test")
    handler = ServiceContextPR(GithubPRHandler("test-owner", "test-repo", "test-token"), llm_config)

    # Mock the requests.get to simulate a 404 error
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPError("404 Client Error: Not Found")

    with patch("httpx.get", return_value=mock_response):
        # Call the method with a non-existent issue reference
        result = handler.strategy.get_context_from_external_issues_references(
            closing_issues=[],
            closing_issue_numbers=[],
            issue_body="This references #999999",  # Non-existent issue
            review_comments=[],
            review_threads=[],
            thread_comments=None,
        )

        # The method should return an empty list since the referenced issue couldn't be fetched
        assert result == []


def test_handle_rate_limit_error() -> None:
    llm_config = LLMConfig(model="test", api_key="test")
    handler = ServiceContextPR(GithubPRHandler("test-owner", "test-repo", "test-token"), llm_config)

    # Mock the requests.get to simulate a rate limit error
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPError("403 Client Error: Rate Limit Exceeded")

    with patch("httpx.get", return_value=mock_response):
        # Call the method with an issue reference
        result = handler.strategy.get_context_from_external_issues_references(
            closing_issues=[],
            closing_issue_numbers=[],
            issue_body="This references #123",
            review_comments=[],
            review_threads=[],
            thread_comments=None,
        )

        # The method should return an empty list since the request was rate limited
        assert result == []


def test_handle_network_error() -> None:
    llm_config = LLMConfig(model="test", api_key="test")
    handler = ServiceContextPR(GithubPRHandler("test-owner", "test-repo", "test-token"), llm_config)

    # Mock the requests.get to simulate a network error
    with patch("httpx.get", side_effect=httpx.NetworkError("Network Error")):
        # Call the method with an issue reference
        result = handler.strategy.get_context_from_external_issues_references(
            closing_issues=[],
            closing_issue_numbers=[],
            issue_body="This references #123",
            review_comments=[],
            review_threads=[],
            thread_comments=None,
        )

        # The method should return an empty list since the network request failed
        assert result == []


def test_successful_issue_reference() -> None:
    llm_config = LLMConfig(model="test", api_key="test")
    handler = ServiceContextPR(GithubPRHandler("test-owner", "test-repo", "test-token"), llm_config)

    # Mock a successful response
    mock_response = MagicMock()
    mock_response.raise_for_status.return_value = None
    mock_response.json.return_value = {"body": "This is the referenced issue body"}

    with patch("httpx.get", return_value=mock_response):
        # Call the method with an issue reference
        result = handler.strategy.get_context_from_external_issues_references(
            closing_issues=[],
            closing_issue_numbers=[],
            issue_body="This references #123",
            review_comments=[],
            review_threads=[],
            thread_comments=None,
        )

        # The method should return a list with the referenced issue body
        assert result == ["This is the referenced issue body"]


class MockLLMResponse:
    """Mock LLM Response class to mimic the actual LLM response structure."""

    class Choice:
        class Message:
            def __init__(self, content: str) -> None:
                self.content = content

        def __init__(self, content: str) -> None:
            self.message = self.Message(content)

    def __init__(self, content: str) -> None:
        self.choices = [self.Choice(content)]


class DotDict(dict):
    """A dictionary that supports dot notation access."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        for key, value in self.items():
            if isinstance(value, dict):
                self[key] = DotDict(value)
            elif isinstance(value, list):
                self[key] = [DotDict(item) if isinstance(item, dict) else item for item in value]

    def __getattr__(self, key: str) -> object:
        if key in self:
            return self[key]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")

    @override
    def __setattr__(self, key: str, value: object) -> None:
        self[key] = value

    @override
    def __delattr__(self, key: str) -> None:
        if key in self:
            del self[key]
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_guess_success_rate_limit_wait_time(mock_litellm_completion: MagicMock, default_config: LLMConfig) -> None:
    """Test that the retry mechanism in guess_success respects wait time between retries."""
    with patch("time.sleep") as mock_sleep:
        # Simulate a rate limit error followed by a successful response
        mock_litellm_completion.side_effect = [
            RateLimitError("Rate limit exceeded", llm_provider="test_provider", model="test_model"),
            DotDict({"choices": [{"message": {"content": "--- success\ntrue\n--- explanation\nRetry successful"}}]}),
        ]

        llm = LLM(config=default_config, service_id="test-service")
        handler = ServiceContextIssue(GithubIssueHandler("test-owner", "test-repo", "test-token"), default_config)
        handler.llm = llm

        # Mock issue and history
        issue = Issue(
            owner="test-owner",
            repo="test-repo",
            number=1,
            title="Test Issue",
            body="This is a test issue.",
            thread_comments=["Please improve error handling"],
        )
        history = [MessageAction(content="Fixed error handling.")]

        # Call guess_success
        success, _, explanation = handler.guess_success(issue, history)

        # Assertions
        assert success is True
        assert explanation == "Retry successful"
        assert mock_litellm_completion.call_count == 2  # Two attempts made
        mock_sleep.assert_called_once()  # Sleep called once between retries

        # Validate wait time
        wait_time = mock_sleep.call_args[0][0]
        assert default_config.retry_min_wait <= wait_time <= default_config.retry_max_wait, (
            f"Expected wait time between {default_config.retry_min_wait} and {default_config.retry_max_wait} seconds, but got {wait_time}"
        )


@patch("openhands.sdk.llm.llm.litellm_completion")
def test_guess_success_exhausts_retries(mock_completion: MagicMock, default_config: LLMConfig) -> None:
    """Test the retry mechanism in guess_success exhausts retries and raises an error."""
    # Simulate persistent rate limit errors by always raising RateLimitError
    mock_completion.side_effect = RateLimitError(
        "Rate limit exceeded", llm_provider="test_provider", model="test_model"
    )

    # Initialize LLM and handler
    llm = LLM(config=default_config, service_id="test-service")
    handler = ServiceContextPR(GithubPRHandler("test-owner", "test-repo", "test-token"), default_config)
    handler.llm = llm

    # Mock issue and history
    issue = Issue(
        owner="test-owner",
        repo="test-repo",
        number=1,
        title="Test Issue",
        body="This is a test issue.",
        thread_comments=["Please improve error handling"],
    )
    history = [MessageAction(content="Fixed error handling.")]

    # Call guess_success and expect it to raise an error after retries
    with pytest.raises(RateLimitError):
        handler.guess_success(issue, history)

    # Assertions
    assert mock_completion.call_count == default_config.num_retries  # Initial call + retries
