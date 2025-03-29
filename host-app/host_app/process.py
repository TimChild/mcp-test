from contextlib import asynccontextmanager

from langchain_core.messages import AnyMessage
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic

from .models import QA, Update

from typing import AsyncIterator, Sequence

from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

model = ChatOpenAI(model="gpt-4o")


@asynccontextmanager
async def connect_client() -> AsyncIterator[MultiServerMCPClient]:
    async with MultiServerMCPClient(
        connections={
            "test-server": SSEConnection(
                transport="sse", url="http://localhost:9090/sse"
            )
        }
    ) as client:
        yield client


async def get_response_updates(
    question: str, message_history: list[QA]
) -> AsyncIterator[Update]:
    """Get the response updates for a question.

    Args:
        question: The question to get the response updates for.
        message_history: The message history.

    Returns:
        The response updates.
    """
    yield Update(type_="ai-delta", delta=f"Question: {question}\n\n")
    yield Update(type_="ai-delta", delta=f"Length History: {len(message_history)}\n\n")

    async with connect_client() as client:
        # NOTE: Tools are loaded immediately after initial connection in the LC implementation
        tools = client.get_tools()
        agent = create_react_agent(model, tools)
        response = await agent.ainvoke({"messages": question})
        validated = AgentStatePydantic.model_validate(response)
        messages = validated.messages
        ai_response = messages[-1].content
        assert isinstance(ai_response, str)
        yield Update(type_="ai-delta", delta=ai_response)
    yield Update(type_="ai-delta", delta="\n\n!!! End of response updates !!!\n\n")
