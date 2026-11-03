from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

from openhands.sdk import LLM, AgentContext, Conversation, LocalWorkspace
from openhands.tools.preset.default import get_default_agent


@pytest.mark.parametrize("model", ["anthropic/claude-sonnet-4-20250514"])
def test_sdk_resolver_stack_constructs(tmp_path: Path, model: str) -> None:
    llm = LLM(model=model, api_key=SecretStr("test-key"), usage_id="test")
    workspace = LocalWorkspace(working_dir=str(tmp_path))
    agent = get_default_agent(llm=llm, cli_mode=True).model_copy(
        update={"agent_context": AgentContext(system_message_suffix="test")}
    )

    conversation: Any = Conversation(
        agent=agent,
        workspace=workspace,
        persistence_dir=None,
        visualizer=None,
        max_iteration_per_run=1,
    )

    assert list(conversation.state.events) == []
    assert conversation.conversation_stats.get_combined_metrics().accumulated_cost == 0
