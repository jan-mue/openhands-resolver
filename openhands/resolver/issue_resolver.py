"""Resolve issues with an OpenHands SDK agent."""

import json
import os
import pathlib
import shutil
import subprocess
import time
from argparse import Namespace
from typing import Any

from pydantic import SecretStr
from termcolor import colored

from openhands.resolver.interfaces.issue import Issue
from openhands.resolver.interfaces.issue_definitions import ServiceContextIssue, ServiceContextPR
from openhands.resolver.issue_handler_factory import IssueHandlerFactory
from openhands.resolver.provider import ProviderType
from openhands.resolver.resolver_output import ResolverOutput
from openhands.resolver.utils import (
    GENERAL_TIMEOUT,
    call_async_from_sync,
    identify_token,
    reset_logger_for_multiprocessing,
)
from openhands.sdk import (
    LLM,
    AgentContext,
    Conversation,
    ImageContent,
    LocalWorkspace,
    Message,
    TextContent,
    get_logger,
)
from openhands.sdk.event.conversation_error import ConversationErrorEvent
from openhands.sdk.llm.exceptions import LLMError
from openhands.tools.preset.default import get_default_agent

logger = get_logger(__name__)

# Don't make this configurable for now, unless we have other competitive agents
AGENT_CLASS = "CodeActAgent"
MIN_REPO_PARTS = 2


def _last_error(events: list[Any]) -> str | None:
    for event in reversed(events):
        if isinstance(event, ConversationErrorEvent):
            return event.detail
    return None


class IssueResolver:
    """Resolve one issue or pull request using an SDK conversation."""

    def __init__(self, args: Namespace) -> None:
        """Initialize the IssueResolver with the given parameters."""
        parts = args.selected_repo.rsplit("/", 1)
        if len(parts) < MIN_REPO_PARTS:
            raise ValueError("Invalid repository format. Expected owner/repo")
        owner, repo = parts

        token = (
            args.token
            or os.getenv("GITHUB_TOKEN")
            or os.getenv("GITLAB_TOKEN")
            or os.getenv("BITBUCKET_TOKEN")
            or os.getenv("AZURE_DEVOPS_TOKEN")
            or os.getenv("FORGEJO_TOKEN")
        )
        username = args.username if args.username else os.getenv("GIT_USERNAME")
        if not username:
            raise ValueError("Username is required.")

        if not token:
            raise ValueError("Token is required.")

        base_domain = getattr(args, "base_domain", None)
        platform = call_async_from_sync(
            identify_token,
            GENERAL_TIMEOUT,
            token,
            base_domain,
        )

        repo_instruction = None
        repo_instruction_file = getattr(args, "repo_instruction_file", None)
        if repo_instruction_file:
            with open(repo_instruction_file) as f:
                repo_instruction = f.read()

        issue_type = args.issue_type

        prompt_file = getattr(args, "prompt_file", None)
        if prompt_file is None:
            if issue_type == "issue":
                prompt_file = os.path.join(os.path.dirname(__file__), "prompts/resolve/basic-with-tests.jinja")
            else:
                prompt_file = os.path.join(os.path.dirname(__file__), "prompts/resolve/basic-followup.jinja")
        with open(prompt_file) as f:
            user_instructions_prompt_template = f.read()

        with open(prompt_file.replace(".jinja", "-conversation-instructions.jinja")) as f:
            conversation_instructions_prompt_template = f.read()

        if base_domain is None:
            base_domain = (
                "github.com"
                if platform == ProviderType.GITHUB
                else "gitlab.com"
                if platform == ProviderType.GITLAB
                else "bitbucket.org"
                if platform == ProviderType.BITBUCKET
                else "bitbucket.example.com"
                if platform == ProviderType.BITBUCKET_DATA_CENTER
                else "codeberg.org"
                if platform == ProviderType.FORGEJO
                else "dev.azure.com"
            )

        model_name = getattr(args, "llm_model", None) or os.getenv("LLM_MODEL") or "anthropic/claude-sonnet-4-20250514"
        api_key = getattr(args, "llm_api_key", None) or os.getenv("LLM_API_KEY")
        if not api_key:
            raise ValueError("LLM_API_KEY is required.")

        base_url = getattr(args, "llm_base_url", None) or os.getenv("LLM_BASE_URL")
        self.llm = LLM(
            model=model_name,
            api_key=SecretStr(api_key),
            base_url=base_url or None,
            usage_id="resolver",
        )

        self.output_dir = args.output_dir
        self.issue_type = issue_type
        self.issue_number = args.issue_number
        self.workspace_base = self.build_workspace_base(self.output_dir, self.issue_type, self.issue_number)
        self.max_iterations = args.max_iterations
        self.owner = owner
        self.repo = repo
        self.platform = platform
        self.user_instructions_prompt_template = user_instructions_prompt_template
        self.conversation_instructions_prompt_template = conversation_instructions_prompt_template
        self.repo_instruction = repo_instruction
        self.comment_id = getattr(args, "comment_id", None)

        factory = IssueHandlerFactory(
            owner=self.owner,
            repo=self.repo,
            token=token,
            username=username,
            platform=self.platform,
            base_domain=base_domain,
            issue_type=self.issue_type,
            llm=self.llm,
        )
        self.issue_handler = factory.create()

    @staticmethod
    def build_workspace_base(output_dir: str, issue_type: str, issue_number: int) -> str:
        """Return the workspace directory for an issue attempt."""
        workspace_base = os.path.join(output_dir, "workspace", f"{issue_type}_{issue_number}")
        return os.path.abspath(workspace_base)

    @staticmethod
    def _run_command(
        command: list[str],
        cwd: str,
        *,
        check: bool = True,
        timeout: float = 300,
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                f"Command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}\n{result.stderr}"
            )
        return result

    def initialize_workspace(self, workspace: LocalWorkspace) -> None:
        """Prepare a cloned workspace for the agent."""
        logger.info("-" * 30)
        logger.info("BEGIN Workspace Initialization Fn")
        logger.info("-" * 30)

        cwd = str(workspace.working_dir)
        self._run_command(["git", "config", "--global", "core.pager", ""], cwd)

        setup_script = os.path.join(cwd, ".openhands", "setup.sh")
        if os.path.exists(setup_script):
            logger.info("Running .openhands/setup.sh")
            self._run_command(["bash", setup_script], cwd, timeout=600)

        hook_script = os.path.join(cwd, ".openhands", "pre-commit.sh")
        hooks_dir = os.path.join(cwd, ".git", "hooks")
        if os.path.exists(hook_script) and os.path.isdir(hooks_dir):
            logger.info("Installing .openhands/pre-commit.sh git hook")
            destination = os.path.join(hooks_dir, "pre-commit")
            shutil.copy2(hook_script, destination)
            os.chmod(destination, 0o700)

    def complete_workspace(self, workspace: LocalWorkspace, base_commit: str) -> dict[str, Any]:
        """Collect the git patch produced by the agent."""
        logger.info("-" * 30)
        logger.info("BEGIN Workspace Completion Fn")
        logger.info("-" * 30)

        cwd = str(workspace.working_dir)
        self._run_command(["git", "add", "-A"], cwd)

        git_patch = None
        for attempt in range(5):
            result = self._run_command(
                ["git", "diff", "--no-color", "--cached", base_commit],
                cwd,
                check=False,
                timeout=600 + 100 * attempt,
            )
            if result.returncode == 0:
                git_patch = result.stdout.strip()
                break
            logger.info("Failed to get git diff, retrying...")
            time.sleep(10)

        if git_patch is None:
            raise RuntimeError("Failed to collect git diff after 5 attempts.")

        logger.info("-" * 30)
        logger.info("END Workspace Completion Fn")
        logger.info("-" * 30)
        return {"git_patch": git_patch}

    async def process_issue(
        self,
        issue: Issue,
        base_commit: str,
        issue_handler: ServiceContextIssue | ServiceContextPR,
        reset_logger: bool = False,
    ) -> ResolverOutput:
        """Run the agent against one issue and return its resolver output."""
        if reset_logger:
            log_dir = os.path.join(self.output_dir, "infer_logs")
            reset_logger_for_multiprocessing(logger, str(issue.number), log_dir)
        else:
            logger.info(f"Starting fixing issue {issue.number}.")

        if os.path.exists(self.workspace_base):
            shutil.rmtree(self.workspace_base)
        shutil.copytree(os.path.join(self.output_dir, "repo"), self.workspace_base)

        workspace = LocalWorkspace(working_dir=self.workspace_base)
        self.initialize_workspace(workspace)

        instruction, conversation_instructions, images_urls = issue_handler.get_instruction(
            issue,
            self.user_instructions_prompt_template,
            self.conversation_instructions_prompt_template,
            self.repo_instruction,
        )

        agent = get_default_agent(llm=self.llm, cli_mode=True)
        if conversation_instructions:
            agent = agent.model_copy(
                update={"agent_context": AgentContext(system_message_suffix=conversation_instructions)}
            )

        conversation = Conversation(
            agent=agent,
            workspace=workspace,
            persistence_dir=None,
            visualizer=None,
            max_iteration_per_run=self.max_iterations,
        )

        content: list[TextContent | ImageContent] = [TextContent(text=instruction)]
        if images_urls:
            content.append(ImageContent(image_urls=images_urls))

        histories: list[dict[str, Any]] = []
        metrics: dict[str, Any] | None = None
        success = False
        comment_success: list[bool] | None = None
        result_explanation = ""
        last_error: str | None = None
        events: list[Any] = []

        try:
            conversation.send_message(Message(role="user", content=content))
            conversation.run()

            events = list(conversation.state.events)
            histories = [event.model_dump(mode="json") for event in events]
            metrics = conversation.conversation_stats.get_combined_metrics().model_dump(mode="json")
            last_error = _last_error(events)
        except (ValueError, RuntimeError, OSError, LLMError) as e:
            last_error = f"Agent failed with error: {e}"
            logger.error(last_error)
            result_explanation = "Agent failed to run"

        return_val = self.complete_workspace(workspace, base_commit)
        git_patch = return_val["git_patch"]
        logger.info(f"Got git diff for instance {issue.number}:\n--------\n{git_patch}\n--------")

        if events:
            try:
                success, comment_success, result_explanation = issue_handler.guess_success(issue, events, git_patch)
            except (ValueError, RuntimeError, OSError, LLMError) as e:
                last_error = f"Failed to evaluate result: {e}"
                logger.error(last_error)
                result_explanation = "Failed to evaluate result"
        else:
            success = False
            comment_success = None
            result_explanation = result_explanation or "Agent failed to run"

        if issue_handler.issue_type == "pr" and comment_success:
            success_log = "I have updated the PR and resolved some of the issues that were cited in the pull request review. Specifically, I identified the following revision requests, and all the ones that I think I successfully resolved are checked off. All the unchecked ones I was not able to resolve, so manual intervention may be required:\n"
            try:
                explanations = json.loads(result_explanation)
            except json.JSONDecodeError:
                logger.error(f"Failed to parse result_explanation as JSON: {result_explanation}")
                explanations = [str(result_explanation)]

            for success_indicator, explanation in zip(comment_success, explanations, strict=False):
                status = colored("[X]", "red") if success_indicator else colored("[ ]", "red")
                bullet_point = colored("-", "yellow")
                success_log += f"\n{bullet_point} {status}: {explanation}"
            logger.info(success_log)

        return ResolverOutput(
            issue=issue,
            issue_type=issue_handler.issue_type,
            instruction=instruction,
            base_commit=base_commit,
            git_patch=git_patch,
            history=histories,
            metrics=metrics,
            success=success,
            comment_success=comment_success,
            result_explanation=result_explanation,
            error=last_error,
        )

    def extract_issue(self) -> Issue:
        """Fetch and return the issue configured for this resolver."""
        issues: list[Issue] = self.issue_handler.get_converted_issues(
            issue_numbers=[self.issue_number], comment_id=self.comment_id
        )

        if not issues:
            raise ValueError(
                f"No issues found for issue number {self.issue_number}. Please verify that:\n"
                f"1. The issue/PR #{self.issue_number} exists in the repository {self.owner}/{self.repo}\n"
                f"2. You have the correct permissions to access it\n"
                f"3. The repository name is spelled correctly"
            )

        return issues[0]

    async def resolve_issue(
        self,
        reset_logger: bool = False,
    ) -> None:
        """Resolve a single issue."""
        issue = self.extract_issue()

        if self.comment_id is not None:
            if (
                self.issue_type == "pr"
                and not issue.review_comments
                and not issue.review_threads
                and not issue.thread_comments
            ):
                raise ValueError(f"Comment ID {self.comment_id} did not have a match for issue {issue.number}")

            if self.issue_type == "issue" and not issue.thread_comments:
                raise ValueError(f"Comment ID {self.comment_id} did not have a match for issue {issue.number}")

        model_name = self.llm.model.split("/")[-1]

        pathlib.Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(self.output_dir, "infer_logs")).mkdir(parents=True, exist_ok=True)
        logger.info(f"Using output directory: {self.output_dir}")

        repo_dir = os.path.join(self.output_dir, "repo")
        if not os.path.exists(repo_dir):
            checkout_output = subprocess.check_output(
                [
                    "git",
                    "clone",
                    self.issue_handler.get_clone_url(),
                    f"{self.output_dir}/repo",
                ]
            ).decode("utf-8")
            if "fatal" in checkout_output:
                raise RuntimeError(f"Failed to clone repository: {checkout_output}")

        base_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_dir,
            )
            .decode("utf-8")
            .strip()
        )
        logger.info(f"Base commit: {base_commit}")

        if self.repo_instruction is None:
            openhands_instructions_path = os.path.join(repo_dir, ".openhands_instructions")
            if os.path.exists(openhands_instructions_path):
                with open(openhands_instructions_path) as f:
                    self.repo_instruction = f.read()

        output_file = os.path.join(self.output_dir, "output.jsonl")
        logger.info(f"Writing output to {output_file}")

        if os.path.exists(output_file):
            with open(output_file) as f:
                for line in f:
                    data = ResolverOutput.model_validate_json(line)
                    if data.issue.number == self.issue_number:
                        logger.warning(f"Issue {self.issue_number} was already processed. Skipping.")
                        return

        with open(output_file, "a") as output_fp:
            logger.info(
                f"Resolving issue {self.issue_number} with Agent {AGENT_CLASS}, model {model_name}, max iterations {self.max_iterations}."
            )

            try:
                if self.issue_type == "pr":
                    branch_to_use = issue.head_branch
                    logger.info(f"Checking out to PR branch {branch_to_use} for issue {issue.number}")

                    if not branch_to_use:
                        raise ValueError("Branch name cannot be None")

                    subprocess.check_output(
                        ["git", "fetch", "origin", branch_to_use],
                        cwd=repo_dir,
                    )
                    subprocess.check_output(
                        ["git", "checkout", branch_to_use],
                        cwd=repo_dir,
                    )

                    base_commit = (
                        subprocess.check_output(
                            ["git", "rev-parse", "HEAD"],
                            cwd=repo_dir,
                        )
                        .decode("utf-8")
                        .strip()
                    )

                output = await self.process_issue(
                    issue,
                    base_commit,
                    self.issue_handler,
                    reset_logger,
                )
                output_fp.write(output.model_dump_json() + "\n")
                output_fp.flush()

            finally:
                logger.info("Finished.")
