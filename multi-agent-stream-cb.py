import json
import asyncio
from distro import name
from os import name
import httpx
import chromadb
from typing import Literal
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# =====================================================================
# 1. THE DATA PLANE (ChromaDB Bootstrap - Kept Stable)
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
# 2. HARDENED SHARED STATE MAP
# =====================================================================
class MultiAgentState(TypedDict):
    messages: list = add_messages
    next_worker: str
    loop_count: int  # 🌟 FIX 1: Track internal workflow loops programmatically

# =====================================================================
# 3. HIGH-SPEED ASYNCHRONOUS STREAMING PROXY CALLER
# =====================================================================
async def call_gateway_proxy_stream(system_prompt: str, messages: list) -> str:
    url = "http://localhost:4000/v1/chat/completions"
    
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if hasattr(msg, "content"): formatted_messages.append({"role": getattr(msg, "type", "user"), "content": msg.content})
        elif isinstance(msg, dict): formatted_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
    payload = {
        "model": "enterprise-brain",
        "messages": formatted_messages,
        "response_format": {"type": "json_object"},
        "stream": True  
    }
    
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
                        print(token, end="", flush=True)
                    except (json.JSONDecodeError, KeyError):
                        continue
                        
    return complete_response_string

# =====================================================================
# 4. MICRO-AGENT NODE ITERATIONS (Harden system role flags)
# =====================================================================
async def supervisor_node(state: MultiAgentState) -> dict:
    print(f"\n🧠 [SUPERVISOR]: Analyzing ledger metrics (Current Graph Loop Count: {state.get('loop_count', 0)})...")
    print("   ↳ [PROXY STREAM]: ", end="", flush=True)
    
    prompt = (
        "You are a Technical SRE Supervisor. Review the history context to assign the next step.\n"
        "Your available worker routing options are:\n"
        "1. 'log_fetcher_agent': Choose this ONLY if logs have not been retrieved from ChromaDB yet.\n"
        "2. 'sre_analyst_agent': Choose this if log details are present in history, but a deep report has not been compiled.\n"
        "3. 'FINISH': Choose this if a comprehensive root-cause analysis report [FINAL SRE REPORT] is already present in the history.\n\n"
        "Respond strictly with a JSON object matching this schema:\n"
        '{"next": "worker_name_or_FINISH"}'
    )
    
    response_str = await call_gateway_proxy_stream(prompt, state["messages"])
    try:
        data = json.loads(response_str.strip())
        next_step = data.get("next", "FINISH")
    except json.JSONDecodeError:
        next_step = "FINISH"
        
    print(f" -> Routing instruction token to: '{next_step}'")
    
    # Increment our circuit breaker tracker loop metric state channel
    current_count = state.get("loop_count", 0)
    return {"next_worker": next_step, "loop_count": current_count + 1}


async def log_fetcher_agent_node(state: MultiAgentState) -> dict:
    print("\n⚙️ [LOG FETCHER AGENT]: Resolving database optimization paths...")
    print("   ↳ [PROXY STREAM]: ", end="", flush=True)
    
    prompt = (
        "You are a specialized retrieval tool agent. Look at history and output an optimized semantic search query "
        "to pull log records from the ChromaDB system layout.\n"
        "Respond strictly with a JSON object following this schema:\n"
        '{"search_query": "search query text"}'
    )
    
    response_str = await call_gateway_proxy_stream(prompt, state["messages"])
    try:
        data = json.loads(response_str.strip())
        search_query = data.get("search_query", "")
    except json.JSONDecodeError:
        search_query = "auth-service loop failure logs"
        
    print(f" -> Querying ChromaDB using parameter: '{search_query}'")
    async with httpx.AsyncClient() as client:
        query_vector = await get_local_embedding_async(client, search_query)
        
    results = GLOBAL_VECTOR_DB.query(query_embeddings=[query_vector], n_results=1)
    retrieved_chunk = results['documents']
    
    # 🌟 FIX 2: Tag as a strict SYSTEM tool log injection so the supervisor reads it contextually
    injection = f"[SYSTEM RETRIEVAL UPDATE] Logs matching runbook: '{retrieved_chunk}'."
    return {"messages": [{"role": "system", "content": injection}]}


async def sre_analyst_agent_node(state: MultiAgentState) -> dict:
    print("\n🔬 [SRE ANALYST AGENT]: Generating comprehensive root-cause analysis report matrix...")
    print("   ↳ [PROXY STREAM]: ", end="", flush=True)
    
    prompt = (
        "You are a Principal Engineer. Review the logs injected in history and output a detailed root-cause analysis report "
        "labeled clearly with [FINAL SRE REPORT] prefix.\n"
        "Respond strictly with a JSON object matching this schema:\n"
        '{"analysis": "Your comprehensive engineering report summary and fix list here"}'
    )
    
    response_str = await call_gateway_proxy_stream(prompt, state["messages"])
    try:
        data = json.loads(response_str.strip())
        final_report = data.get("analysis", "")
    except json.JSONDecodeError:
        final_report = "Analysis data stream interrupted."
        
    print(" -> Report compilation finalized.")
    return {"messages": [{"role": "assistant", "content": f"[FINAL SRE REPORT]:\n{final_report}"}]}

# =====================================================================
# 5. CONTROL PLANE HARDENED GRAPH WIRE ROUTER
# =====================================================================
def supervisor_router(state: MultiAgentState) -> Literal["log_fetcher_agent_node", "sre_analyst_agent_node", "__end__"]:
    # A completed report is terminal, regardless of the supervisor's next token.
    for msg in reversed(state["messages"]):
        content = msg.content if hasattr(msg, "content") else msg.get("content", "")
        if "[FINAL SRE REPORT]:" in content:
            print(f"\n🤖 {content}\n")
            return "__end__"

    # 🌟 FIX 3: THE GRAPH CIRCUIT BREAKER INTERCEPT
    # If the system has fanned out too many times, intercept and terminate gracefully
    if state.get("loop_count", 0) >= 4:
        print("\n🚨 [GRAPH CIRCUIT BREAKER ACTIVATED]: Maximum multi-agent task fanning threshold crossed. Forcing safe system graceful termination to shield hardware and token limits...")
        
        # Output the analyst report context directly from state history if present
        for msg in reversed(state["messages"]):
            content = msg.content if hasattr(msg, "content") else msg.get("content", "")
            if "[FINAL SRE REPORT]:" in content:
                print(f"\n🤖 {content}\n")
                return "__end__"
        return "__end__"

    target = state.get("next_worker")
    if target == "log_fetcher_agent": return "log_fetcher_agent_node"
    elif target == "sre_analyst_agent": return "sre_analyst_agent_node"
    else:
        print("\n✅ [SUPERVISOR]: Task goal achieved successfully. Ending execution loop workflows.")
        last_msg = state["messages"][-1]
        last_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
        if "[FINAL SRE REPORT]:" in last_content:
            print(f"\n🤖 {last_content}\n")
        return "__end__"

# =====================================================================
# 6. GRAPH REFACTORING COMPILATION
# =====================================================================
builder = StateGraph(MultiAgentState)

builder.add_node("supervisor_node", supervisor_node)
builder.add_node("log_fetcher_agent_node", log_fetcher_agent_node)
builder.add_node("sre_analyst_agent_node", sre_analyst_agent_node)

builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", supervisor_router)
builder.add_edge("log_fetcher_agent_node", "supervisor_node")
builder.add_edge("sre_analyst_agent_node", "supervisor_node")
app = builder.compile()

async def main():
    initial_input = {"messages": [{"role": "user", "content": "The auth-service is failing token validation loops. Why?"}],"loop_count": 0}
    print("👤 [USER]: The auth-service is failing token validation loops. Why?")
    await app.ainvoke(initial_input)
    
if __name__ == "__main__":
    asyncio.run(main())