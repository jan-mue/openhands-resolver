"""Data model for a single resolver result."""

from typing import Any

from pydantic import BaseModel

from openhands.resolver.interfaces.issue import Issue


class ResolverOutput(BaseModel):
    """Result produced after attempting to resolve an issue."""

    issue: Issue
    issue_type: str
    instruction: str
    base_commit: str
    git_patch: str
    history: list[dict[str, Any]]
    metrics: dict[str, Any] | None
    success: bool
    comment_success: list[bool] | None
    result_explanation: str
    error: str | None
