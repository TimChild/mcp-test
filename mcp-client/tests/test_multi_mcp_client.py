from mcp_client import MultiMCPClient


def test_init_multi_mcp_client():
    multi_mcp_client = MultiMCPClient(connections={})
    assert isinstance(multi_mcp_client, MultiMCPClient)
