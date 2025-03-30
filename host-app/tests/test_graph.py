from typing import Any

import pytest
from langgraph.graph.graph import CompiledGraph

from host_app.graph import GraphRunner, InputState, OutputState, make_graph


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
