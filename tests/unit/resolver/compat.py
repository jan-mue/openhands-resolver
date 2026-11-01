"""Small compatibility helpers for tests ported from the V0 resolver."""

import random
import time
from typing import Any

from litellm.exceptions import RateLimitError
from pydantic import SecretStr

import openhands.sdk.llm.llm as sdk_llm_module
from openhands.sdk import Message, TextContent


class LLM:
    def __init__(
        self,
        config: "LLMConfig | None" = None,
        service_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.model = kwargs.pop("model", None) or getattr(config, "model", "test")
        self.api_key = kwargs.pop("api_key", None) or getattr(config, "api_key", "test")
        self.base_url = kwargs.pop("base_url", None) or getattr(config, "base_url", None)
        self.num_retries = kwargs.pop("num_retries", getattr(config, "num_retries", 0))
        self.retry_min_wait = kwargs.pop("retry_min_wait", getattr(config, "retry_min_wait", 0))
        self.retry_max_wait = kwargs.pop("retry_max_wait", getattr(config, "retry_max_wait", 0))
        self.service_id = service_id

    def completion(self, messages: list[Any], **kwargs: Any) -> Any:
        converted = []
        for message in messages:
            if isinstance(message, Message):
                converted.append(message)
            elif isinstance(message, dict):
                converted.append(
                    Message(
                        role=message.get("role", "user"),
                        content=[TextContent(text=message.get("content", ""))],
                    )
                )
            else:
                converted.append(message)

        api_key = self.api_key
        if isinstance(api_key, SecretStr):
            api_key = api_key.get_secret_value()

        attempts = max(1, self.num_retries)
        for attempt in range(attempts):
            try:
                return sdk_llm_module.litellm_completion(
                    model=self.model,
                    messages=converted,
                    api_key=api_key,
                    base_url=self.base_url,
                    **kwargs,
                )
            except RateLimitError:
                if attempt == attempts - 1:
                    raise
                time.sleep(random.uniform(self.retry_min_wait, self.retry_max_wait))
        raise AssertionError("unreachable")


class LLMConfig(LLM):
    def __init__(
        self,
        model: str = "test",
        api_key: str | SecretStr | None = "test",
        base_url: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=base_url,
            **kwargs,
        )


class MessageAction:
    def __init__(self, content: str = "", image_urls: list[str] | None = None, **_: Any):
        self.content = content
        self.message = content
        self.source = "agent"
        self.image_urls = image_urls or []


def message_text(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "".join(getattr(part, "text", "") for part in content)
