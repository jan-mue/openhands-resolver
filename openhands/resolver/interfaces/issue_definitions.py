import json
import os
import re
from collections.abc import Sequence
from typing import Any, ClassVar

import jinja2

from openhands.resolver.interfaces.issue import (
    Issue,
    IssueHandlerInterface,
    ReviewThread,
)
from openhands.resolver.utils import extract_image_urls
from openhands.sdk import Message, MessageEvent, TextContent
from openhands.sdk.llm import content_to_str


def _message_text(event: object) -> str | None:
    if isinstance(event, MessageEvent):
        if event.source != "agent":
            return None
        return "".join(content_to_str(event.llm_message.content)).strip()

    # Support lightweight test doubles and legacy-style objects.
    if getattr(event, "source", "agent") != "agent":
        return None

    message = getattr(event, "message", None)
    if isinstance(message, str):
        return message.strip()

    content = getattr(event, "content", None)
    if isinstance(content, str):
        return content.strip()

    return None


def _last_agent_message(history: Sequence[object]) -> str:
    for event in reversed(history):
        text = _message_text(event)
        if text is not None:
            return text
    return ""


def _response_text(response: object) -> str:
    choices = getattr(response, "choices", None)
    if choices:
        return choices[0].message.content.strip()

    message = getattr(response, "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if content is not None:
        return "".join(content_to_str(content)).strip()

    raise ValueError(f"Unsupported LLM response: {response!r}")


class ServiceContext:
    issue_type: ClassVar[str]
    default_git_patch: ClassVar[str] = "No changes made yet"

    def __init__(self, strategy: IssueHandlerInterface, llm: Any = None):
        self._strategy = strategy
        self.llm: Any = llm

    def set_strategy(self, strategy: IssueHandlerInterface) -> None:
        self._strategy = strategy


# Strategy context interface
class ServiceContextPR(ServiceContext):
    issue_type: ClassVar[str] = "pr"

    def __init__(self, strategy: IssueHandlerInterface, llm: Any):
        super().__init__(strategy, llm)

    def get_clone_url(self) -> str:
        return self._strategy.get_clone_url()

    def download_issues(self) -> list[Any]:
        return self._strategy.download_issues()

    def guess_success(
        self,
        issue: Issue,
        history: Sequence[object],
        git_patch: str | None = None,
    ) -> tuple[bool, list[bool] | None, str]:
        """Guess if the issue is fixed based on the history, issue description and git patch.

        Args:
            issue: The issue to check
            history: The agent's history
            git_patch: Optional git patch showing the changes made
        """
        last_message = _last_agent_message(history)

        issues_context = json.dumps(issue.closing_issues, indent=4)
        success_list = []
        explanation_list = []

        # Handle PRs with file-specific review comments
        if issue.review_threads:
            for review_thread in issue.review_threads:
                if issues_context and last_message:
                    success, explanation = self._check_review_thread(
                        review_thread, issues_context, last_message, git_patch
                    )
                else:
                    success, explanation = False, "Missing context or message"
                success_list.append(success)
                explanation_list.append(explanation)
        # Handle PRs with only thread comments (no file-specific review comments)
        elif issue.thread_comments:
            if issue.thread_comments and issues_context and last_message:
                success, explanation = self._check_thread_comments(
                    issue.thread_comments, issues_context, last_message, git_patch
                )
            else:
                success, explanation = (
                    False,
                    "Missing thread comments, context or message",
                )
            success_list.append(success)
            explanation_list.append(explanation)
        elif issue.review_comments:
            # Handle PRs with only review comments (no file-specific review comments or thread comments)
            if issue.review_comments and issues_context and last_message:
                success, explanation = self._check_review_comments(
                    issue.review_comments, issues_context, last_message, git_patch
                )
            else:
                success, explanation = (
                    False,
                    "Missing review comments, context or message",
                )
            success_list.append(success)
            explanation_list.append(explanation)
        else:
            # No review comments, thread comments, or file-level review comments found
            return False, None, "No feedback was found to process"

        # Return overall success (all must be true) and explanations
        if not success_list:
            return False, None, "No feedback was processed"
        return all(success_list), success_list, json.dumps(explanation_list)

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        return self._strategy.get_converted_issues(issue_numbers, comment_id)

    def get_instruction(
        self,
        issue: Issue,
        user_instructions_prompt_template: str,
        conversation_instructions_prompt_template: str,
        repo_instruction: str | None = None,
    ) -> tuple[str, str, list[str]]:
        """Generate instruction for the agent."""
        user_instruction_template = jinja2.Template(user_instructions_prompt_template)
        conversation_instructions_template = jinja2.Template(conversation_instructions_prompt_template)
        images = []

        issues_str = None
        if issue.closing_issues:
            issues_str = json.dumps(issue.closing_issues, indent=4)
            images.extend(extract_image_urls(issues_str))

        # Handle PRs with review comments
        review_comments_str = None
        if issue.review_comments:
            review_comments_str = json.dumps(issue.review_comments, indent=4)
            images.extend(extract_image_urls(review_comments_str))

        # Handle PRs with file-specific review comments
        review_thread_str = None
        review_thread_file_str = None
        if issue.review_threads:
            review_threads = [review_thread.comment for review_thread in issue.review_threads]
            review_thread_files = []
            for review_thread in issue.review_threads:
                review_thread_files.extend(review_thread.files)
            review_thread_str = json.dumps(review_threads, indent=4)
            review_thread_file_str = json.dumps(review_thread_files, indent=4)
            images.extend(extract_image_urls(review_thread_str))

        # Format thread comments if they exist
        thread_context = ""
        if issue.thread_comments:
            thread_context = "\n---\n".join(issue.thread_comments)
            images.extend(extract_image_urls(thread_context))

        user_instruction = user_instruction_template.render(
            review_comments=review_comments_str,
            review_threads=review_thread_str,
            files=review_thread_file_str,
            thread_context=thread_context,
        )

        conversation_instructions = conversation_instructions_template.render(
            issues=issues_str, repo_instruction=repo_instruction
        )

        return user_instruction, conversation_instructions, images

    def _check_feedback_with_llm(self, prompt: str) -> tuple[bool, str]:
        """Helper function to check feedback with LLM and parse response."""
        if self.llm is None:
            raise RuntimeError("An LLM is required to evaluate feedback.")

        response = self.llm.completion(messages=[Message(role="user", content=[TextContent(text=prompt)])])
        answer = _response_text(response)
        pattern = r"--- success\n*(true|false)\n*--- explanation*\n((?:.|\n)*)"
        match = re.search(pattern, answer)
        if match:
            return match.group(1).lower() == "true", match.group(2).strip()
        return False, f"Failed to decode answer from LLM response: {answer}"

    def _check_review_thread(
        self,
        review_thread: ReviewThread,
        issues_context: str,
        last_message: str,
        git_patch: str | None = None,
    ) -> tuple[bool, str]:
        """Check if a review thread's feedback has been addressed."""
        files_context = json.dumps(review_thread.files, indent=4)

        with open(
            os.path.join(
                os.path.dirname(__file__),
                "../prompts/guess_success/pr-feedback-check.jinja",
            ),
        ) as f:
            template = jinja2.Template(f.read())

        prompt = template.render(
            issue_context=issues_context,
            feedback=review_thread.comment,
            files_context=files_context,
            last_message=last_message,
            git_patch=git_patch or self.default_git_patch,
        )

        return self._check_feedback_with_llm(prompt)

    def _check_thread_comments(
        self,
        thread_comments: list[str],
        issues_context: str,
        last_message: str,
        git_patch: str | None = None,
    ) -> tuple[bool, str]:
        """Check if thread comments feedback has been addressed."""
        thread_context = "\n---\n".join(thread_comments)

        with open(
            os.path.join(
                os.path.dirname(__file__),
                "../prompts/guess_success/pr-thread-check.jinja",
            ),
        ) as f:
            template = jinja2.Template(f.read())

        prompt = template.render(
            issue_context=issues_context,
            thread_context=thread_context,
            last_message=last_message,
            git_patch=git_patch or self.default_git_patch,
        )

        return self._check_feedback_with_llm(prompt)

    def _check_review_comments(
        self,
        review_comments: list[str],
        issues_context: str,
        last_message: str,
        git_patch: str | None = None,
    ) -> tuple[bool, str]:
        """Check if review comments feedback has been addressed."""
        review_context = "\n---\n".join(review_comments)

        with open(
            os.path.join(
                os.path.dirname(__file__),
                "../prompts/guess_success/pr-review-check.jinja",
            ),
        ) as f:
            template = jinja2.Template(f.read())

        prompt = template.render(
            issue_context=issues_context,
            review_context=review_context,
            last_message=last_message,
            git_patch=git_patch or self.default_git_patch,
        )

        return self._check_feedback_with_llm(prompt)


class ServiceContextIssue(ServiceContext):
    issue_type: ClassVar[str] = "issue"

    def __init__(self, strategy: IssueHandlerInterface, llm: Any = None):
        super().__init__(strategy, llm)

    def get_base_url(self) -> str:
        return self._strategy.get_base_url()

    def get_branch_url(self, branch_name: str) -> str:
        return self._strategy.get_branch_url(branch_name)

    def get_download_url(self) -> str:
        return self._strategy.get_download_url()

    def get_clone_url(self) -> str:
        return self._strategy.get_clone_url()

    def get_graphql_url(self) -> str:
        return self._strategy.get_graphql_url()

    def get_headers(self) -> dict[str, str]:
        return self._strategy.get_headers()

    def get_authorize_url(self) -> str:
        return self._strategy.get_authorize_url()

    def get_pull_url(self, pr_number: int) -> str:
        return self._strategy.get_pull_url(pr_number)

    def get_compare_url(self, branch_name: str) -> str:
        return self._strategy.get_compare_url(branch_name)

    def download_issues(self) -> list[Any]:
        return self._strategy.download_issues()

    def get_branch_name(
        self,
        base_branch_name: str,
    ) -> str:
        return self._strategy.get_branch_name(base_branch_name)

    def branch_exists(self, branch_name: str) -> bool:
        return self._strategy.branch_exists(branch_name)

    def get_default_branch_name(self) -> str:
        return self._strategy.get_default_branch_name()

    def create_pull_request(self, data: dict[str, Any] | None = None) -> dict[str, Any]:
        if data is None:
            data = {}
        return self._strategy.create_pull_request(data)

    def request_reviewers(self, reviewer: str, pr_number: int) -> None:
        return self._strategy.request_reviewers(reviewer, pr_number)

    def reply_to_comment(self, pr_number: int, comment_id: str, reply: str) -> None:
        return self._strategy.reply_to_comment(pr_number, comment_id, reply)

    def send_comment_msg(self, issue_number: int, msg: str) -> None:
        return self._strategy.send_comment_msg(issue_number, msg)

    def get_issue_comments(self, issue_number: int, comment_id: int | None = None) -> list[str] | None:
        return self._strategy.get_issue_comments(issue_number, comment_id)

    def get_instruction(
        self,
        issue: Issue,
        user_instructions_prompt_template: str,
        conversation_instructions_prompt_template: str,
        repo_instruction: str | None = None,
    ) -> tuple[str, str, list[str]]:
        """Generate instruction for the agent."""
        # Format thread comments if they exist
        thread_context = ""
        if issue.thread_comments:
            thread_context = "\n\nIssue Thread Comments:\n" + "\n---\n".join(issue.thread_comments)

        images = []
        images.extend(extract_image_urls(issue.body))
        images.extend(extract_image_urls(thread_context))

        user_instructions_template = jinja2.Template(user_instructions_prompt_template)
        user_instructions = user_instructions_template.render(
            body=issue.title + "\n\n" + issue.body + thread_context
        )  # Issue body and comments

        conversation_instructions_template = jinja2.Template(conversation_instructions_prompt_template)
        conversation_instructions = conversation_instructions_template.render(
            repo_instruction=repo_instruction,
        )

        return user_instructions, conversation_instructions, images

    def guess_success(
        self, issue: Issue, history: Sequence[object], git_patch: str | None = None
    ) -> tuple[bool, list[bool] | None, str]:
        """Guess if the issue is fixed based on the history and the issue description.

        Args:
            issue: The issue to check
            history: The agent's history
            git_patch: Optional git patch showing the changes made
        """
        last_message = _last_agent_message(history)
        # Include thread comments in the prompt if they exist
        issue_context = issue.body
        if issue.thread_comments:
            issue_context += "\n\nIssue Thread Comments:\n" + "\n---\n".join(issue.thread_comments)

        with open(
            os.path.join(
                os.path.dirname(__file__),
                "../prompts/guess_success/issue-success-check.jinja",
            ),
        ) as f:
            template = jinja2.Template(f.read())
        prompt = template.render(
            issue_context=issue_context,
            last_message=last_message,
            git_patch=git_patch or self.default_git_patch,
        )

        if self.llm is None:
            raise RuntimeError("An LLM is required to evaluate feedback.")

        response = self.llm.completion(messages=[Message(role="user", content=[TextContent(text=prompt)])])
        answer = _response_text(response)
        pattern = r"--- success\n*(true|false)\n*--- explanation*\n((?:.|\n)*)"
        match = re.search(pattern, answer)
        if match:
            return match.group(1).lower() == "true", None, match.group(2)

        return False, None, f"Failed to decode answer from LLM response: {answer}"

    def get_converted_issues(
        self, issue_numbers: list[int] | None = None, comment_id: int | None = None
    ) -> list[Issue]:
        return self._strategy.get_converted_issues(issue_numbers, comment_id)
