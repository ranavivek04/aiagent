import os
import asyncio
import sys
from mcp.server.mcpserver import MCPServer

# =====================================================================
# 1. INITIALIZE THE PROTOCOL GATEWAY SERVER
# =====================================================================
# We establish an architectural MCP Server wrapper node
server = MCPServer(
    name="enterprise-runbook-mcp-server",
    version="1.0.0",
)

# Define the local host operating system directory limit boundary we want to expose securely
TARGET_LOG_DIR = os.path.abspath("./")

# =====================================================================
# 2. REGISTER SERVICE RESOURCES (Exposing Data Schemas)
# =====================================================================
@server.resource(
    "file://knowledge_base.txt",
    name="Production Operations System Runbook Documentation",
    description="Contains active root-cause mappings for connection and gateway cluster errors.",
    mime_type="text/plain",
)
async def handle_read_resource() -> str:
    """Secure data stream accessor endpoint fetching text records on demand."""
    print(
        "[MCP SERVER]: Streaming target resource bytes across the protocol wire "
        "for: 'file://knowledge_base.txt'",
        flush=True,
    )
    
    file_path = os.path.join(TARGET_LOG_DIR, "knowledge_base.txt")
    if not os.path.exists(file_path):
        return "System Runbook file data layer is currently unpopulated on disk."
        
    with open(file_path, "r") as f:
        return f.read()

# =====================================================================
# 3. REGISTER PROTOCOL RUNTIME UTILITIES (Exposing Executable Tools)
# =====================================================================
@server.tool(
    name="query_live_directory",
    description="Scans the localized workspace to verify active log filenames present on disk.",
)
async def handle_execute_tool() -> str:
    """Executes local host operating system actions on behalf of the agent."""
    print("[MCP SERVER]: Executing system utility tool 'query_live_directory'...", flush=True)
    files = os.listdir(TARGET_LOG_DIR)
    return f"Active files present inside the working directory block: {str(files)}"

# =====================================================================
# 4. INITIALIZE HIGH-PERFORMANCE ASYNCHRONOUS TRANSPORT
# =====================================================================
async def main():
    print(
        "[MCP SERVER]: Protocol service mesh successfully mounted over stdio transport.",
        file=sys.stderr,
        flush=True,
    )
    await server.run_stdio_async()

if __name__ == "__main__":
    asyncio.run(main())
