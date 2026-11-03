"""Factory for creating provider-specific issue handlers."""

from openhands.resolver.interfaces.azure_devops import AzureDevOpsIssueHandler
from openhands.resolver.interfaces.bitbucket import BitbucketIssueHandler, BitbucketPRHandler
from openhands.resolver.interfaces.bitbucket_data_center import BitbucketDCIssueHandler, BitbucketDCPRHandler
from openhands.resolver.interfaces.forgejo import ForgejoIssueHandler, ForgejoPRHandler
from openhands.resolver.interfaces.github import GithubIssueHandler, GithubPRHandler
from openhands.resolver.interfaces.gitlab import GitlabIssueHandler, GitlabPRHandler
from openhands.resolver.interfaces.issue_definitions import LLMClient, ServiceContextIssue, ServiceContextPR
from openhands.resolver.provider import ProviderType

MIN_OWNER_PARTS = 2


class IssueHandlerFactory:
    """Create the service context for a provider and issue type."""

    def __init__(
        self,
        owner: str,
        repo: str,
        token: str,
        username: str,
        platform: ProviderType,
        base_domain: str,
        issue_type: str,
        llm: LLMClient | None,
    ) -> None:
        """Store the parameters used to create a handler."""
        self.owner = owner
        self.repo = repo
        self.token = token
        self.username = username
        self.platform = platform
        self.base_domain = base_domain
        self.issue_type = issue_type
        self.llm = llm

    def create(self) -> ServiceContextIssue | ServiceContextPR:  # noqa: PLR0911
        """Create the handler for the configured provider and issue type."""
        if self.issue_type == "issue":
            if self.platform == ProviderType.GITHUB:
                return ServiceContextIssue(
                    GithubIssueHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm,
                )
            if self.platform == ProviderType.GITLAB:
                return ServiceContextIssue(
                    GitlabIssueHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm,
                )
            if self.platform == ProviderType.BITBUCKET:
                return ServiceContextIssue(
                    BitbucketIssueHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm,
                )
            if self.platform == ProviderType.BITBUCKET_DATA_CENTER:
                return ServiceContextIssue(
                    BitbucketDCIssueHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm,
                )
            if self.platform == ProviderType.FORGEJO:
                return ServiceContextIssue(
                    ForgejoIssueHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm,
                )
            if self.platform == ProviderType.AZURE_DEVOPS:
                # Parse owner as organization/project
                parts = self.owner.split("/")
                if len(parts) < MIN_OWNER_PARTS:
                    raise ValueError(
                        f"Invalid Azure DevOps owner format: {self.owner}. Expected format: organization/project"
                    )

                organization = parts[0]
                project = parts[1]

                return ServiceContextIssue(
                    AzureDevOpsIssueHandler(
                        self.token,
                        organization,
                        project,
                        self.repo,
                    ),
                    self.llm,
                )
            raise ValueError(f"Unsupported platform: {self.platform}")
        if self.issue_type == "pr":
            if self.platform == ProviderType.GITHUB:
                return ServiceContextPR(
                    GithubPRHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm,
                )
            if self.platform == ProviderType.GITLAB:
                return ServiceContextPR(
                    GitlabPRHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm,
                )
            if self.platform == ProviderType.BITBUCKET:
                return ServiceContextPR(
                    BitbucketPRHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm,
                )
            if self.platform == ProviderType.BITBUCKET_DATA_CENTER:
                return ServiceContextPR(
                    BitbucketDCPRHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm,
                )
            if self.platform == ProviderType.FORGEJO:
                return ServiceContextPR(
                    ForgejoPRHandler(
                        self.owner,
                        self.repo,
                        self.token,
                        self.username,
                        self.base_domain,
                    ),
                    self.llm,
                )
            if self.platform == ProviderType.AZURE_DEVOPS:
                # Parse owner as organization/project
                parts = self.owner.split("/")
                if len(parts) < MIN_OWNER_PARTS:
                    raise ValueError(
                        f"Invalid Azure DevOps owner format: {self.owner}. Expected format: organization/project"
                    )

                organization = parts[0]
                project = parts[1]

                # For now, use the same handler for both issues and PRs
                return ServiceContextPR(
                    AzureDevOpsIssueHandler(
                        self.token,
                        organization,
                        project,
                        self.repo,
                    ),
                    self.llm,
                )
            raise ValueError(f"Unsupported platform: {self.platform}")
        raise ValueError(f"Invalid issue type: {self.issue_type}")
