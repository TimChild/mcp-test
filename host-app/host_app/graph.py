"""Regular graph version of langgraph."""

import logging
import uuid
from typing import AsyncIterator, Literal

from dependency_injector.wiring import Provide, inject
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.runnables.schema import EventData
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient, SSEConnection
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.func import entrypoint, task
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Adapters, Application, LLMs

from .models import GraphUpdate, UpdateTypes


class GraphRunner:
    def __init__(
        self, mcp_client: MultiMCPClient = Provide[Application.adapters.mcp_client]
    ) -> None:
        self.mcp_client = mcp_client
        self.graph: CompiledGraph = make_graph()

    async def astream_events(
        self, input: BaseModel, thread_id: str | None = None
    ) -> AsyncIterator[GraphUpdate]:
        """Run the graph, yield events converted to GraphUpdates."""
        thread_id = thread_id or str(uuid.uuid4())
        yield GraphUpdate(type_=UpdateTypes.graph_start, data=thread_id)
        async for event in self.graph.astream_events(
            input=input,
            config={"configurable": {"thread_id": thread_id}},
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
                        yield GraphUpdate(type_=UpdateTypes.ai_delta, delta=content)
                case "on_chat_model_end":
                    chunk = event_data.get("output", None)
                    assert isinstance(chunk, AIMessage)
                    yield GraphUpdate(type_=UpdateTypes.ai_message_end)
                case "on_tool_start":
                    chunk = event_data.get("input", None)
                    assert isinstance(chunk, dict)
                    yield GraphUpdate(type_=UpdateTypes.tool_start, name=event["name"], data=chunk)
                case "on_tool_end":
                    chunk = event_data.get("output", None)
                    assert isinstance(chunk, ToolMessage)
                    yield GraphUpdate(
                        type_=UpdateTypes.tool_end, name=event["name"], data=str(chunk.content)
                    )
                case _:
                    logging.debug(f"Ignoring event: {event_type}")
        yield GraphUpdate(type_=UpdateTypes.graph_end)


class FullState(BaseModel):
    question: str
    response_messages: list[AIMessage | ToolMessage] = []


class InputState(BaseModel):
    question: str


class OutputState(BaseModel):
    response_messages: list[AIMessage | ToolMessage]


SYSTEM_PROMPT = """
You are a chatbot operating in a developer debugging environment. You can give detailed information about any information you have access to (you do not have to worry about hiding implementation details from a user).
Respond in markdown.
"""


# async def process(inputs: InputState, mcp_client: MultiMCPClient = Provide[Adapters.mcp_client], chat_model: BaseChatModel = Provide[LLMs.main_model]) -> Command[Literal["__end__", "tool_node"]]:
@inject
async def process(
    inputs: InputState,
    mcp_client: MultiMCPClient = Provide[Application.adapters.mcp_client],
    chat_model: BaseChatModel = Provide[Application.llms.main_model],
) -> OutputState:
    responses: list[AIMessage | ToolMessage] = []
    question = inputs.question
    logging.debug(f"Processing question: {question}")

    async with mcp_client as client:
        tools = await client.get_tools()
        model = chat_model.bind_tools(tools)

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
            results = await ToolNode(tools=tools, name="tool_node").ainvoke(input=response)
            print(type(results))
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
    update = OutputState(response_messages=[AIMessage(content=f"Received: {inputs.question}")])
    return update
    # return Command(update=update, goto=["tool_caller_node", "sub_assistant_caller_node"])


def make_graph(checkpointer: None = None) -> CompiledGraph:
    graph = StateGraph(state_schema=FullState)

    graph.add_node("process", process)
    # graph.add_node("tool_node", ToolNode)
    graph.set_entry_point("process")

    compiled_graph = graph.compile(
        checkpointer=checkpointer, interrupt_before=None, interrupt_after=None, debug=True
    )
    return compiled_graph
