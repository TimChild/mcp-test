from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


class MultiClient:
    def __init__(self, lc_client: MultiServerMCPClient) -> None:
        self.lc_client = lc_client

    async def __aenter__(self) -> "MultiClient":
        """Connects to all servers during context."""
        self.lc_client = await self.lc_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Closes all server connections."""
        await self.lc_client.__aexit__(exc_type, exc_value, traceback)

    async def get_tools(self) -> list[BaseTool]:
        """Get all tools available from all connected servers."""
        # NOTE: lc loads on initial connection, so don't need to await here (in general it would be awaited though)
        return self.lc_client.get_tools()
