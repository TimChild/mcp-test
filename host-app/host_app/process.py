from contextlib import asynccontextmanager
from .models import QA, Update

from typing import AsyncIterator

from mcp_client.client import MCPClient
from mcp_client.client import Agent


@asynccontextmanager
async def connect_client() -> AsyncIterator[MCPClient]:
    client = MCPClient()
    try:
        await client.connect_to_server("http://localhost:9090/sse")
        yield client
    finally:
        await client.cleanup()


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
        agent = Agent(client.session)
        response = await agent.process_query(question)
        yield Update(type_="ai-delta", delta=response)
    yield Update(type_="ai-delta", delta="\n\n!!! End of response updates !!!\n\n")
