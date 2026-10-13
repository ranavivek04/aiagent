import json
import asyncio
import sys
from typing import Literal

# Import the core MCP Client connection utilities
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: list = add_messages

# =====================================================================
# UNIVERSAL CORE MCP AGENT GRAPH ORCHESTRATOR
# =====================================================================
async def run_mcp_agent_workflow():
    print("🔌 Initializing Master MCP Client Session Controller...")
    
    # Configure the background sub-process command launching your standalone server script
    # It passes data strings back and forth across python standard streams safely
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["mcp_server.py"],
    )
    
    async with stdio_client(server_params) as (read_channel, write_channel):
        async with ClientSession(read_channel, write_channel) as session:
            # Step A: Perform the formal JSON-RPC handshaking connection validation
            await session.initialize()
            print("✅ [MCP CLIENT]: Handshake complete. Connected securely to custom MCP Server.")
            
            # Step B: Dynamically pull down available data structures from the remote server interface
            resources = await session.list_resources()
            tools = await session.list_tools()
            
            print(f"📊 [MCP METADATA CAPTURED]:")
            print(f"   ↳ Discovered Remote Resources: {[r.name for r in resources.resources]}")
            print(f"   ↳ Discovered Remote Tools:     {[t.name for t in tools.tools]}")
            
            # Step C: Instruct our LangGraph node to fetch the data directly using the abstract protocol
            async def mcp_data_ingestion_node(state: AgentState) -> dict:
                print("\n🔄 [NODE]: Agent Node executing an abstract MCP network fetch look up...")
                
                # We request the target file data purely over its URI standard handle string name!
                target_resource_uri = "file://knowledge_base.txt"
                content_payload = await session.read_resource(target_resource_uri)
                
                print(f"📥 [DATA CONTEXT INGESTED VIA MCP]:")
                print(f"   {content_payload.contents[0].text[:120]}...")
                
                # Execute the live directory search tool discovered over the network mapping
                print("\n⚙️ [NODE]: Triggering live remote tool call via MCP protocol layer...")
                tool_result = await session.call_tool("query_live_directory", arguments={})
                print(f"🛠️ [TOOL RESULT INSIDE GRAPH]: {tool_result.content[0].text}")
                
                return {"messages": [{"role": "assistant", "content": "MCP Pipeline validation verified healthy."}]}

            # Build and compile a clean testing graph
            workflow = StateGraph(AgentState)
            workflow.add_node("mcp_data_ingestion_node", mcp_data_ingestion_node)
            workflow.add_edge(START, "mcp_data_ingestion_node")
            app = workflow.compile()
            
            print("\n🚀 Invoking LangGraph runtime engine...")
            await app.ainvoke({"messages": [{"role": "user", "content": "Trigger pipeline diagnostics check."}]})

if __name__ == "__main__":
    asyncio.run(run_mcp_agent_workflow())
