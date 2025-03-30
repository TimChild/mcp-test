"""Regular graph version of langgraph."""

import logging
import uuid
from typing import AsyncIterator

from dependency_injector.wiring import Provide
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage
from langchain_core.runnables.schema import EventData
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Adapters

from .models import GraphUpdate, UpdateTypes


class GraphRunner:
    def __init__(self, mcp_client: MultiMCPClient = Provide[Adapters.mcp_client]) -> None:
        self.mcp_client = mcp_client
        self.graph: CompiledGraph = make_graph()

    async def astream_events(
        self, input: BaseModel, thread_id: str | None = None
    ) -> AsyncIterator[GraphUpdate]:
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


def process(inputs: InputState) -> OutputState:
    return OutputState(response_messages=[AIMessage(content=f"Received: {inputs.question}")])


def make_graph(checkpointer: None = None) -> CompiledGraph:
    graph = StateGraph(state_schema=FullState)

    graph.add_node("process", process)

    graph.set_entry_point("process")
    graph.set_finish_point("process")
    compiled_graph = graph.compile(
        checkpointer=checkpointer, interrupt_before=None, interrupt_after=None, debug=True
    )
    return compiled_graph
