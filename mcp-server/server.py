from mcp.server.fastmcp import FastMCP

SERVER_PORT = 9090
SERVER_TRANSPORT = "sse"


mcp = FastMCP("test-server", host="localhost", port=SERVER_PORT)


@mcp.tool(
    name="test-tool",
    description="A test tool description",
)
async def get_data() -> dict[str, str]:
    return {"data": "Hello World!"}


@mcp.resource(
    uri="data://example-{name}",
    name="Get name resource",
    description="Test resource description",
    mime_type="application/json",
)
async def get_resource(name: str) -> dict[str, str]:
    return {"name": name}


if __name__ == "__main__":
    print("Starting server...")
    mcp.run(transport=SERVER_TRANSPORT)

print("exiting script")
