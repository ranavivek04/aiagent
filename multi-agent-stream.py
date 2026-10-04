import json
import asyncio
import httpx
import chromadb
from typing import Literal
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# =====================================================================
# 1. THE DATA PLANE (ChromaDB Bootstrap)
# =====================================================================
async def get_local_embedding_async(client: httpx.AsyncClient, text: str) -> list:
    url = "http://localhost:11434/api/embed"
    payload = {"model": "mxbai-embed-large", "input": text}
    response = await client.post(url, json=payload, timeout=30.0)
    result = response.json()
    embeddings = result['embeddings']
    return embeddings[0] if embeddings and isinstance(embeddings[0], list) else embeddings

def initialize_vector_store():
    chroma_client = chromadb.Client()
    try: chroma_client.delete_collection(name="multi_agent_stream_knowledge")
    except Exception: pass
    collection = chroma_client.create_collection(name="multi_agent_stream_knowledge")
    with open("knowledge_base.txt", "r") as f: content = f.read()
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    
    import urllib.request
    url = "http://localhost:11434/api/embed"
    for index, chunk in enumerate(chunks):
        payload = {"model": "mxbai-embed-large", "input": chunk}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            embeddings = result['embeddings']
            vector = embeddings[0] if embeddings and isinstance(embeddings[0], list) else embeddings
            collection.add(embeddings=[vector], documents=[chunk], ids=[f"doc_{index}"])
    return collection

GLOBAL_VECTOR_DB = initialize_vector_store()

# =====================================================================
# 2. SHARED STATE MAP
# =====================================================================
class MultiAgentState(TypedDict):
    messages: list = add_messages
    next_worker: str

# =====================================================================
# 3. 🛡️ HARDENED ASYNCHRONOUS STREAMING PROXY CALLER
# =====================================================================
async def call_gateway_proxy_stream(system_prompt: str, messages: list) -> str:
    """Hits the local LiteLLM Proxy using strict timeouts and non-blocking token streaming."""
    url = "http://localhost:4000/v1/chat/completions"
    
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if hasattr(msg, "content"): formatted_messages.append({"role": getattr(msg, "type", "user"), "content": msg.content})
        elif isinstance(msg, dict): formatted_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
    payload = {
        "model": "enterprise-brain",
        "messages": formatted_messages,
        "response_format": {"type": "json_object"},
        "stream": True  # 🌟 FIX 1: Turn streaming back on for our multi-agent hops
    }
    
    # 🌟 FIX 2: Reinstate a strict, defensive 10-second read timeout ceiling!
    # Because data blocks flow continuously, the read timer refreshes on every chunk packet.
    custom_timeout = httpx.Timeout(30.0, connect=5.0)
    complete_response_string = ""
    
    async with httpx.AsyncClient(timeout=custom_timeout) as client:
        async with client.stream("POST", url, json=payload, headers={"Authorization": "Bearer local-key"}) as stream:
            async for line in stream.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk_json = json.loads(data_str)
                        token = chunk_json['choices'][0]['delta'].get('content', '')
                        complete_response_string += token
                        
                        # Print characters as they arrive so the user tracking console remains highly active
                        print(token, end="", flush=True)
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
    return complete_response_string

# =====================================================================
# 4. MICRO-AGENT NODE ITERATIONS
# =====================================================================
async def supervisor_node(state: MultiAgentState) -> dict:
    print("\n🧠 [SUPERVISOR]: Analyzing ledger metrics...")
    print("   ↳ [PROXY STREAM]: ", end="", flush=True)
    
    prompt = (
        "You are a Technical SRE Supervisor. Your job is to assign the next step to your workers.\n"
        "Your available workers are:\n"
        "1. 'log_fetcher_agent': Use this worker if logs haven't been fetched yet or if you need to query the database runbooks.\n"
        "2. 'sre_analyst_agent': Use this worker if logs have been successfully retrieved and you need a deep engineering root-cause analysis.\n"
        "3. 'FINISH': Use this token when the SRE analyst has successfully delivered a clear final answer to the user.\n\n"
        "Respond with a JSON object following this exact schema:\n"
        '{"next": "worker_name_or_FINISH"}'
    )
    
    response_str = await call_gateway_proxy_stream(prompt, state["messages"])
    try:
        data = json.loads(response_str.strip())
        next_step = data.get("next", "FINISH")
    except json.JSONDecodeError:
        next_step = "FINISH"
        
    print(f" -> Chosen target: '{next_step}'")
    return {"next_worker": next_step}


async def log_fetcher_agent_node(state: MultiAgentState) -> dict:
    print("\n⚙️ [LOG FETCHER AGENT]: Resolving database optimization paths...")
    print("   ↳ [PROXY STREAM]: ", end="", flush=True)
    
    prompt = (
        "You are a specialized data retrieval agent. Read the conversation history and generate an optimized semantic "
        "search query string to lookup the system logs database runbook file.\n"
        "Respond strictly with a JSON object following this schema:\n"
        '{"search_query": "your descriptive search query here"}'
    )
    
    response_str = await call_gateway_proxy_stream(prompt, state["messages"])
    try:
        data = json.loads(response_str.strip())
        search_query = data.get("search_query", "")
    except json.JSONDecodeError:
        search_query = "auth-service failure"
        
    print(f" -> Querying ChromaDB using parameter: '{search_query}'")
    async with httpx.AsyncClient() as client:
        query_vector = await get_local_embedding_async(client, search_query)
        
    results = GLOBAL_VECTOR_DB.query(query_embeddings=[query_vector], n_results=1)
    retrieved_chunk = results['documents']
    
    injection = (
        f"[LOG FETCHER SYSTEM UPDATE]: Fetched matching architectural context chunk from ChromaDB: '{retrieved_chunk}'. "
        "Task completed. Passing execution control back to Supervisor."
    )
    return {"messages": [{"role": "user", "content": injection}]}


async def sre_analyst_agent_node(state: MultiAgentState) -> dict:
    print("\n🔬 [SRE ANALYST AGENT]: Generating comprehensive root-cause analysis report matrix...")
    print("   ↳ [PROXY STREAM]: ", end="", flush=True)
    
    prompt = (
        "You are a Principal Distributed Systems Engineer. Review the retrieved logs injected into history and "
        "compile a detailed, thorough root-cause analysis and mitigation action items roadmap layout.\n"
        "Respond strictly with a JSON object matching this schema:\n"
        '{"analysis": "Your comprehensive engineering report summary and fix list here"}'
    )
    
    response_str = await call_gateway_proxy_stream(prompt, state["messages"])
    try:
        data = json.loads(response_str.strip())
        final_report = data.get("analysis", "")
    except json.JSONDecodeError:
        final_report = "Analysis malformed during parsing routines."
        
    print(" -> Report compilation finalized.")
    return {"messages": [{"role": "assistant", "content": f"[FINAL SRE REPORT]:\n{final_report}"}]}

# =====================================================================
# 5. CONTROL PLANE GRAPH WIRE LAYOUT
# =====================================================================
def supervisor_router(state: MultiAgentState) -> Literal["log_fetcher_agent_node", "sre_analyst_agent_node", "__end__"]:
    target = state.get("next_worker")
    if target == "log_fetcher_agent": return "log_fetcher_agent_node"
    elif target == "sre_analyst_agent": return "sre_analyst_agent_node"
    else:
        last_msg = state["messages"][-1]
        last_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
        if "[FINAL SRE REPORT]:" in last_content:
            print(f"\n🤖 {last_content}\n")
        return "__end__"

builder = StateGraph(MultiAgentState)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("log_fetcher_agent_node", log_fetcher_agent_node)
builder.add_node("sre_analyst_agent_node", sre_analyst_agent_node)

builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", supervisor_router)
builder.add_edge("log_fetcher_agent_node", "supervisor_node")
builder.add_edge("sre_analyst_agent_node", "supervisor_node")

app = workflow = builder.compile()

async def main():
    initial_input = {"messages": [{"role": "user", "content": "The auth-service is failing token validation loops. Why?"}]}
    print("👤 [USER]: The auth-service is failing token validation loops. Why?")
    await app.ainvoke(initial_input)

if __name__ == "__main__":
    asyncio.run(main())
