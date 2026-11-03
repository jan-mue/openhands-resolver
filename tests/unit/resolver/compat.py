"""Small compatibility helpers for tests ported from the V0 resolver."""

import random
import time

from litellm.exceptions import RateLimitError
from pydantic import SecretStr

import openhands.sdk.llm.llm as sdk_llm_module
from openhands.sdk import Message, TextContent


class LLM:
    def __init__(
        self,
        config: "LLMConfig | None" = None,
        service_id: str | None = None,
        *,
        model: str | None = None,
        api_key: str | SecretStr | None = None,
        base_url: str | None = None,
        num_retries: int = 0,
        retry_min_wait: float = 0,
        retry_max_wait: float = 0,
    ) -> None:
        self.model = model or (config.model if config else "test")
        self.api_key = api_key or (config.api_key if config else "test")
        self.base_url = base_url or (config.base_url if config else None)
        self.num_retries = num_retries or (config.num_retries if config else 0)
        self.retry_min_wait = retry_min_wait or (config.retry_min_wait if config else 0)
        self.retry_max_wait = retry_max_wait or (config.retry_max_wait if config else 0)
        self.service_id = service_id

    def completion(self, messages: list[Message], **kwargs: object) -> object:
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
    pass


class MessageAction:
    def __init__(self, content: str = "", image_urls: list[str] | None = None, **_: object) -> None:
        self.content = content
        self.message = content
        self.source = "agent"
        self.image_urls = image_urls or []


def message_text(message: object) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, (list, tuple)):
        return "".join(str(getattr(part, "text", "")) for part in content)
    return str(content)
