import asyncio
import logging
from typing import Any
from dataclasses import dataclass
from contextlib import asynccontextmanager
import uuid
from langchain_core.runnables.schema import EventData, StreamEvent
from langchain_core.tools import BaseTool
from pydantic import BaseModel

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    SystemMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
    get_buffer_string,
)

from .models import QA, Update, UpdateTypes

from typing import AsyncIterator, reveal_type

from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from langchain_openai import ChatOpenAI

from langgraph.func import entrypoint, task
from langgraph.checkpoint.memory import MemorySaver

SYSTEM_PROMPT = """
You are a chatbot operating in a developer debugging environment. You can give detailed information about any information you have access to (you do not have to worry about hiding implementation details from a user).
Respond in markdown.
"""


@asynccontextmanager
async def connect_client() -> AsyncIterator[MultiServerMCPClient]:
    async with MultiServerMCPClient(
        connections={"test-server": SSEConnection(transport="sse", url="http://localhost:9090/sse")}
    ) as client:
        yield client


async def get_response_updates(question: str, message_history: list[QA]) -> AsyncIterator[Update]:
    """Get the response updates for a question.

    Args:
        question: The question to get the response updates for.
        message_history: The message history.

    Returns:
        The response updates.
    """
    yield Update(type_=UpdateTypes.start, data=f"Question: {question}\n\n")
    yield Update(type_=UpdateTypes.preprocess, data=f"Length History: {len(message_history)}\n\n")

    async for event in process.astream_events(
        input=InputState(question=question),
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    ):
        event_type = event["event"]
        event_data: EventData = event["data"]
        match event_type:
            case "on_chat_model_stream":
                chunk = event_data.get("chunk", None)
                if chunk:
                    assert isinstance(chunk, AIMessageChunk)
                    content = chunk.content
                    assert isinstance(content, str)
                    yield Update(type_=UpdateTypes.ai_delta, delta=content)
            case "on_chat_model_end":
                chunk = event_data.get("output", None)
                assert isinstance(chunk, AIMessage)
                yield Update(type_=UpdateTypes.ai_message_end)
            case "on_tool_start":
                chunk = event_data.get("input", None)
                assert isinstance(chunk, dict)
                yield Update(type_=UpdateTypes.tool_start, name=event["name"], data=chunk)
            case "on_tool_end":
                chunk = event_data.get("output", None)
                assert isinstance(chunk, ToolMessage)
                yield Update(
                    type_=UpdateTypes.tool_end, name=event["name"], data=str(chunk.content)
                )
            case _:
                print(f"Ignoring event: {event_type}")

    yield Update(type_=UpdateTypes.end, data="\n\n!!! End of response updates !!!\n\n")


checkpointer = MemorySaver()


class InputState(BaseModel):
    question: str


@task
async def call_tool(tool_call: ToolCall, tools: list[BaseTool]) -> ToolMessage:
    logging.critical(f"Calling tool: {tool_call}")
    tool = next(tool for tool in tools if tool.name == tool_call["name"])
    tool_call_result = await tool.ainvoke(tool_call)
    logging.critical(f"Tool call result: {tool_call_result}")
    assert isinstance(tool_call_result, ToolMessage)
    return tool_call_result
    # tool_call_result: CallToolResult = await client.sessions["test-server"].call_tool(
    #     tool_call["name"], tool_call["args"]
    # )
    # if tool_call_result.isError:
    #     return ToolMessage(
    #         tool_call_id=tool_call["id"],
    #         content=f"Tool errored: {str(tool_call_result.content)}",
    #     )
    # return ToolMessage(
    #     tool_call_id=tool_call["id"], content=str(tool_call_result.content)
    # )


@task
def fixed_val(inp: str) -> str:
    return inp + "fixed"


@dataclass
class OutputState:
    response_messages: list[AIMessage | ToolMessage]
    # response_messages: list[BaseMessage]


@entrypoint(checkpointer=checkpointer)
async def process(inputs: InputState) -> OutputState:
    responses: list[AIMessage | ToolMessage] = []
    question = inputs.question
    model = ChatOpenAI(model="gpt-4o")
    logging.debug(f"Processing question: {question}")

    async with connect_client() as client:
        # NOTE: Tools are loaded immediately after initial connection in the LC implementation
        tools = client.get_tools()
        model = model.bind_tools(tools)

        messages: list[BaseMessage] = [
            SystemMessage(SYSTEM_PROMPT),
            HumanMessage(question),
        ]
        response: BaseMessage = await model.ainvoke(input=messages)
        assert isinstance(response, AIMessage)
        responses.append(response)
        messages.append(response)
        logging.debug("Got initial response")

        assert isinstance(response, AIMessage)
        if response.tool_calls:
            logging.debug("Calling tools")
            # futures = [call_tool(tool_call, tools) for tool_call in response.tool_calls]
            # logging.critical(f"Futures: {futures}")
            # results = await asyncio.gather(*futures)
            # logging.critical(f"Results: {results}")
            # if any(not isinstance(result, ToolMessage) for result in results):
            #     logging.error(f"Got invalid tool response: {results}")
            assert len(response.tool_calls) == 1
            # results = [await call_tool(response.tool_calls[0], tools)]
            results = [
                ToolMessage(
                    tool_call_id=response.tool_calls[0]["id"], content=fixed_val("bla").result()
                )
            ]

            responses.extend(results)
            messages.extend(results)
            logging.debug("Got tool responses")
            try:
                response = await model.ainvoke(input=messages)
            except Exception as e:
                logging.error(f"Error invoking model: {e}")
                logging.error(f"Messages: {messages}")
                raise e
            assert isinstance(response, AIMessage)
            responses.append(response)

        logging.debug("Returning responses")
        # return responses
        return OutputState(response_messages=responses)
