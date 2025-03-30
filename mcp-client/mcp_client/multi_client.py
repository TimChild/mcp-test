import asyncio

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import (
    MultiServerMCPClient,
    SSEConnection,
    StdioConnection,
)
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession


class LCClientPatch(MultiServerMCPClient):
    initialize_timeout_s: float = 5

    # added timeout on intiaializing a session
    async def _initialize_session_and_load_tools(
        self, server_name: str, session: ClientSession
    ) -> None:
        """Initialize a session and load tools from it.

        Args:
            server_name: Name to identify this server connection
            session: The ClientSession to initialize
        """
        # Initialize the session
        try:
            await asyncio.wait_for(session.initialize(), timeout=self.initialize_timeout_s)
        except asyncio.TimeoutError:
            raise RuntimeError("Failed to initialize session within timeout")
        self.sessions[server_name] = session

        # Load tools from this server
        server_tools = await load_mcp_tools(session)
        self.server_name_to_tools[server_name] = server_tools


class MultiMCPClient:
    def __init__(self, connections: dict[str, SSEConnection | StdioConnection]) -> None:
        """Initializes an adapter for multiple mcp clients.

        Args:
            connections: A dictionary mapping server names to connection configurations.
                Each configuration can be either a StdioConnection or SSEConnection.
        """
        self.connections = connections
        self.lc_client = LCClientPatch(connections=connections)

    async def __aenter__(self) -> "MultiMCPClient":
        """Connects to all servers during context."""
        self.lc_client = await self.lc_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback) -> None:  # noqa: ANN001
        """Closes all server connections."""
        await self.lc_client.__aexit__(exc_type, exc_value, traceback)

    async def get_tools(self) -> list[BaseTool]:
        """Get all tools available from all connected servers."""
        # NOTE: lc loads on initial connection, so don't need to await here (in general it would be awaited though)
        async with self.lc_client:
            return self.lc_client.get_tools()
