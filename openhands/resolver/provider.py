from enum import StrEnum


class ProviderType(StrEnum):
    """Git hosting providers supported by the resolver."""

    GITHUB = "github"
    GITLAB = "gitlab"
    BITBUCKET = "bitbucket"
    BITBUCKET_DATA_CENTER = "bitbucket_data_center"
    AZURE_DEVOPS = "azure_devops"
    FORGEJO = "forgejo"
