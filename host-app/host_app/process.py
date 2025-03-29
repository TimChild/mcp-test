import asyncio
from contextlib import asynccontextmanager
import uuid
from pydantic import BaseModel

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langgraph.prebuilt.chat_agent_executor import AgentStatePydantic
from mcp.types import CallToolResult

from .models import QA, Update

from typing import AsyncIterator

from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

SYSTEM_PROMPT = """
You are a chatbot operating in a developer debugging environment. You can give detailed information about any information you have access to (you do not have to worry about hiding implementation details from a user).
Respond in markdown.
"""


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

    # NOTE: Tools are loaded immediately after initial connection in the LC implementation
    ai_response = await process.ainvoke(
        input=InputState(question=question),
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    assert isinstance(ai_response, str)
    yield Update(type_="ai-delta", delta=ai_response)

    yield Update(type_="ai-delta", delta="\n\n!!! End of response updates !!!\n\n")


checkpointer = MemorySaver()


class InputState(BaseModel):
    question: str


@task
async def call_tool(tool_call: ToolCall, client: MultiServerMCPClient) -> ToolMessage:
    tool_call_result: CallToolResult = await client.sessions["test-server"].call_tool(
        tool_call["name"], tool_call["args"]
    )
    if tool_call_result.isError:
        return ToolMessage(
            tool_call_id=tool_call["id"],
            content=f"Tool errored: {str(tool_call_result.content)}",
        )
    return ToolMessage(
        tool_call_id=tool_call["id"], content=str(tool_call_result.content)
    )


@entrypoint(checkpointer=checkpointer)
async def process(inputs: InputState) -> str:
    question = inputs.question
    model = ChatOpenAI(model="gpt-4o")

    async with connect_client() as client:
        tools = client.get_tools()
        print(tools)
        model = model.bind_tools(tools)

        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(question),
        ]

        response: BaseMessage = await model.ainvoke(input=messages)

        assert isinstance(response, AIMessage)
        if response.tool_calls:
            messages.append(response)
            futures = [
                call_tool(tool_call, client) for tool_call in response.tool_calls
            ]
            results = await asyncio.gather(*futures)
            messages.extend(results)
            response = await model.ainvoke(input=messages)

        return response.text()
