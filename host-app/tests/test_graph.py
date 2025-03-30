from typing import Any

import pytest
from langgraph.graph.graph import CompiledGraph

from host_app.graph import GraphRunner, InputState, OutputState, make_graph
from host_app.models import GraphUpdate


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


@pytest.fixture
def graph_runner() -> GraphRunner:
    return GraphRunner()


async def test_astream_graph_runner(graph_runner: GraphRunner):
    updates: list[GraphUpdate] = []
    async for update in graph_runner.astream_events(input=InputState(question="Hello")):
        assert isinstance(update, GraphUpdate)
        updates.append(update)

    assert len(updates) > 0
