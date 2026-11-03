"""Shared utilities for the resolver."""

import asyncio
import base64
import concurrent.futures
import logging
import multiprocessing as mp
import os
import re
from collections.abc import Awaitable, Callable
from typing import cast

import httpx

from openhands.resolver.provider import ProviderType
from openhands.sdk import get_logger

logger = get_logger(__name__)

GENERAL_TIMEOUT = 60
HTTP_OK = 200
PASSWD_UID_INDEX = 2


def httpx_verify_option() -> bool:
    """Return the TLS verification setting for HTTP clients."""
    value = os.getenv("DISABLE_SSL_VERIFY", "").strip().lower()
    return value not in {"1", "true", "yes", "on"}


def call_async_from_sync[T](
    async_fn: Callable[..., Awaitable[T]], timeout: float, *args: object, **kwargs: object
) -> T:
    """Run ``async_fn`` from synchronous code, even when a loop is running."""

    async def _run() -> T:
        return await asyncio.wait_for(async_fn(*args, **kwargs), timeout=timeout)

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_run())

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return cast(T, executor.submit(asyncio.run, _run()).result())


async def _check_http_token(
    client: httpx.AsyncClient,
    url: str,
    headers: dict[str, str],
) -> bool:
    try:
        response = await client.get(url, headers=headers)
    except httpx.HTTPError:
        return False
    return response.status_code == HTTP_OK


async def identify_token(token: str, base_domain: str | None) -> ProviderType:
    """Identify which git hosting provider a token belongs to.

    Args:
        token: The personal access token to check.
        base_domain: Custom base domain for a self-hosted provider.

    Raises:
        ValueError: If the token is not accepted by any known provider.
    """
    base_url = f"https://{base_domain}" if base_domain else None
    custom = bool(
        base_domain
        and base_domain
        not in {
            "github.com",
            "gitlab.com",
            "bitbucket.org",
            "codeberg.org",
            "dev.azure.com",
        }
    )

    candidates: list[tuple[ProviderType, str, dict[str, str]]] = []

    if not custom or base_domain in {None, "github.com"}:
        candidates.append(
            (
                ProviderType.GITHUB,
                "https://api.github.com/user",
                {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
            )
        )

    if not custom or base_domain in {None, "gitlab.com"}:
        candidates.append(
            (
                ProviderType.GITLAB,
                "https://gitlab.com/api/v4/user",
                {"Authorization": f"Bearer {token}"},
            )
        )

    if not custom or base_domain in {None, "bitbucket.org"}:
        candidates.append(
            (
                ProviderType.BITBUCKET,
                "https://api.bitbucket.org/2.0/user",
                {"Authorization": f"Bearer {token}"},
            )
        )

    if not custom or base_domain in {None, "codeberg.org"}:
        candidates.append(
            (
                ProviderType.FORGEJO,
                "https://codeberg.org/api/v1/user",
                {"Authorization": f"token {token}"},
            )
        )

    if not custom or base_domain in {None, "dev.azure.com"}:
        azure_auth = base64.b64encode(f":{token}".encode()).decode()
        candidates.append(
            (
                ProviderType.AZURE_DEVOPS,
                "https://app.vssps.visualstudio.com/_apis/profile/profiles/me?api-version=7.0",
                {"Authorization": f"Basic {azure_auth}"},
            )
        )

    if base_url:
        basic_auth = base64.b64encode(token.encode() if ":" in token else f"x-token:{token}".encode()).decode()
        candidates.extend(
            [
                (
                    ProviderType.GITLAB,
                    f"{base_url}/api/v4/user",
                    {"Authorization": f"Bearer {token}"},
                ),
                (
                    ProviderType.GITHUB,
                    f"{base_url}/api/v3/user",
                    {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"},
                ),
                (
                    ProviderType.FORGEJO,
                    f"{base_url}/api/v1/user",
                    {"Authorization": f"token {token}"},
                ),
                (
                    ProviderType.BITBUCKET_DATA_CENTER,
                    f"{base_url}/rest/api/1.0/users?limit=1",
                    {"Authorization": f"Basic {basic_auth}"},
                ),
            ]
        )

    async with httpx.AsyncClient(
        timeout=10.0,
        follow_redirects=True,
        verify=httpx_verify_option(),
    ) as client:
        for provider, url, headers in candidates:
            if await _check_http_token(client, url, headers):
                return provider

    raise ValueError("Token is invalid.")


def cleanup() -> None:
    """Terminate child processes left behind by the resolver."""
    logger.info("Cleaning up child processes...")
    for process in mp.active_children():
        logger.info(f"Terminating child process: {process.name}")
        process.terminate()
        process.join()


def reset_logger_for_multiprocessing(logger: logging.Logger, instance_id: str, log_dir: str) -> None:
    """Redirect logging to a per-instance file while keeping console output."""
    log_file = os.path.join(log_dir, f"instance_{instance_id}.log")

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(console_handler)
    logger.info(
        f"Starting resolver for instance {instance_id}.\n"
        f'Hint: run "tail -f {log_file}" to see live logs in a separate shell'
    )

    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)


def extract_image_urls(issue_body: str) -> list[str]:
    """Extract image URLs from Markdown image tags in an issue body."""
    # Regular expression to match Markdown image syntax ![alt text](image_url)
    image_pattern = r"!\[.*?\]\((https?://[^\s)]+)\)"
    return re.findall(image_pattern, issue_body)


def extract_issue_references(body: str) -> list[int]:
    """Extract issue numbers referenced from an issue body."""
    # First, remove code blocks as they may contain false positives
    body = re.sub(r"```.*?```", "", body, flags=re.DOTALL)

    # Remove inline code
    body = re.sub(r"`[^`]*`", "", body)

    # Remove URLs that contain hash symbols
    body = re.sub(r"https?://[^\s)]*#\d+[^\s)]*", "", body)

    pattern = r"(?:^|[\s\[({]|[^\w#])#(\d+)(?=[\s,.\])}]|$)"
    return [int(match) for match in re.findall(pattern, body)]


def get_unique_uid(start_uid: int = 1000) -> int:
    """Return an unused UID starting from ``start_uid``."""
    existing_uids = set()
    with open("/etc/passwd") as passwd_file:
        for line in passwd_file:
            parts = line.split(":")
            if len(parts) > PASSWD_UID_INDEX:
                try:
                    existing_uids.add(int(parts[PASSWD_UID_INDEX]))
                except ValueError:
                    continue

    while start_uid in existing_uids:
        start_uid += 1

    return start_uid
