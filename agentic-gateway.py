import json
import asyncio
import httpx
from typing import Literal
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: list = add_messages

async def gateway_analyzer_node(state: AgentState) -> dict:
    print("\n🔄 [NODE]: Analyzer shifting routing execution plane to Gateway Proxy...")
    print("🌊 [PROXY STREAM STARTING]: ", end="", flush=True)
    
    # 🌟 ARCHITECTURAL INFRASTRUCTURE SHIFT
    # We point to port 4000 (LiteLLM Proxy) instead of hitting raw Ollama directly!
    url = "http://localhost:4000/v1/chat/completions"
    
    # Standardized system layout schema payload accepted by OpenAI/LiteLLM specs
    payload = {
        "model": "enterprise-brain",  # 🌟 Targets the abstract gateway model string name!
        "messages": [
            {
                "role": "system",
                "content": "You are a production engineering agent. Respond with a JSON object containing a key named 'status' set to 'healthy'."
            }
        ] + [{"role": msg.get("role", "user"), "content": msg.get("content", "")} for msg in state["messages"]],
        "response_format": {"type": "json_object"},
        "stream": True # Streaming tokens via the proxy layer
    }
    
    complete_response = ""
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload, headers={"Authorization": "Bearer local-key"}, timeout=30.0) as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        token = chunk_json['choices'][0]['delta'].get('content', '')
                        complete_response += token
                        print(token, end="", flush=True)
                    except json.JSONDecodeError:
                        continue
                        
    print(" 🌊 [PROXY STREAM COMPLETED]")
    return {"messages": [{"role": "assistant", "content": complete_response}]}

# Setup Workflow Core Topology
workflow = StateGraph(AgentState)
workflow.add_node("gateway_analyzer_node", gateway_analyzer_node)
workflow.add_edge(START, "gateway_analyzer_node")
app = workflow.compile()

async def main():
    initial_input = {"messages": [{"role": "user", "content": "Ping system health check parameters."}]}
    await app.ainvoke(initial_input)

if __name__ == "__main__":
    asyncio.run(main())
