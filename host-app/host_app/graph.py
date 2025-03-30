"""Regular graph version of langgraph."""

from dependency_injector.wiring import Provide
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph
from langgraph.graph.graph import CompiledGraph
from mcp_client import MultiMCPClient
from pydantic import BaseModel

from host_app.containers import Adapters


class GraphRunner:
    def __init__(self, mcp_client: MultiMCPClient = Provide[Adapters.mcp_client]) -> None:
        self.mcp_client = mcp_client
        self.graph: CompiledGraph = make_graph()


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
