from typing import Any

import pytest
from dependency_injector.wiring import Provide, inject
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.graph import CompiledGraph
from mcp_client import MultiMCPClient

from host_app.containers import Adapters, Application
from host_app.graph import GraphRunner, InputState, OutputState, make_graph
from host_app.models import GraphUpdate, UpdateTypes


@pytest.fixture(autouse=True)
def container() -> Application:
    container = Application()
    container.config.from_yaml("config.yml")
    container.wire(modules=["host_app.graph"])
    return container


def test_compile_graph():
    graph = make_graph()
    assert isinstance(graph, CompiledGraph)


@pytest.fixture(scope="module")
def graph() -> CompiledGraph:
    return make_graph()


async def test_invoke_graph(graph: CompiledGraph):
    result: dict[str, Any] = await graph.ainvoke(input=InputState(question="Hello"))

    validated: OutputState = OutputState.model_validate(result)
    assert validated.response_messages[0].content == "Received: Hello"


def test_init_graph_runner():
    runner = GraphRunner()
    assert isinstance(runner, GraphRunner)
    connections = runner.mcp_client.connections
    assert isinstance(connections, dict)
    assert list(connections.keys()) == ["example_server"]


@pytest.fixture
def graph_runner() -> GraphRunner:
    return GraphRunner()


async def test_astream_graph_runner(graph_runner: GraphRunner):
    updates: list[GraphUpdate] = []
    async for update in graph_runner.astream_events(input=InputState(question="Hello")):
        assert isinstance(update, GraphUpdate)
        updates.append(update)

    assert len(updates) > 0
    assert updates[0].type_ == UpdateTypes.graph_start
    assert updates[-1].type_ == UpdateTypes.graph_end
