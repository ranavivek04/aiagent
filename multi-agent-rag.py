import json
import asyncio
import httpx
import chromadb
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# =====================================================================
# 1. THE DATA PLANE (ChromaDB Core Setup)
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
    try: chroma_client.delete_collection(name="multi_agent_knowledge")
    except Exception: pass
    collection = chroma_client.create_collection(name="multi_agent_knowledge")
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
    next_worker: str # 🌟 Control plane token directing who executes next

# =====================================================================
# 3. HIGH-SPEED ASYNCHRONOUS ENGINE PROXY CALLER
# =====================================================================
async def call_gateway_proxy(system_prompt: str, messages: list) -> str:
    """Helper to hit our local LiteLLM Gateway Proxy at port 4000."""
    url = "http://localhost:4000/v1/chat/completions"
    
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if hasattr(msg, "content"): formatted_messages.append({"role": getattr(msg, "type", "user"), "content": msg.content})
        elif isinstance(msg, dict): formatted_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
    payload = {
        "model": "enterprise-brain",
        "messages": formatted_messages,
        "response_format": {"type": "json_object"},
        "stream": False # Set to False for easier clean multi-agent data handoffs
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload, headers={"Authorization": "Bearer local-key"}, timeout=None)
        result = response.json()
        return result['choices'][0]['message']['content']

# =====================================================================
# 4. HYPER-SPECIALISED AGENT NODES
# =====================================================================

async def supervisor_node(state: MultiAgentState) -> dict:
    """Node 1: The Traffic Cop.

    Orchestrates control-flow handoffs between workers.
    """
    print("🧠 [SUPERVISOR]: Reviewing operational ledger to assign tasks...")
    
    prompt = (
        "You are a Technical SRE Supervisor. Your job is to assign the next step to your workers.\n"
        "Your available workers are:\n"
        "1. 'log_fetcher_agent': Use this worker if logs haven't been fetched yet or if you need to query the database runbooks.\n"
        "2. 'sre_analyst_agent': Use this worker if logs have been successfully retrieved and you need a deep engineering root-cause analysis.\n"
        "3. 'FINISH': Use this token when the SRE analyst has successfully delivered a clear final answer to the user.\n\n"
        "Respond with a JSON object following this exact schema:\n"
        '{"next": "worker_name_or_FINISH"}'
    )
    
    response_str = await call_gateway_proxy(prompt, state["messages"])
    data = json.loads(response_str.strip())
    next_step = data.get("next", "FINISH")
    
    print(f"   ↳ [DECISION]: Routing execution token directly to: '{next_step}'")
    return {"next_worker": next_step}


async def log_fetcher_agent_node(state: MultiAgentState) -> dict:
    """Node 2: Specialized Data Worker.

    Natively coupled to ChromaDB search.
    """
    print("⚙️ [LOG FETCHER AGENT]: Compiling database search parameter optimization string...")
    
    prompt = (
        "You are a specialized data retrieval agent. Read the conversation history and generate an optimized semantic "
        "search query string to lookup the system logs database runbook file.\n"
        "Respond strictly with a JSON object following this schema:\n"
        '{"search_query": "your descriptive search query here"}'
    )
    
    response_str = await call_gateway_proxy(prompt, state["messages"])
    data = json.loads(response_str.strip())
    search_query = data.get("search_query", "")
    
    print(f"   ↳ [DB LOOKUP]: Querying local ChromaDB space for conceptual matches on '{search_query}'...")
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
    """Node 3: Specialized Reasoning Worker."""
    print("🔬 [SRE ANALYST AGENT]: Running deep vector compilation matrix routines...")
    
    prompt = (
        "You are a Principal Distributed Systems Engineer. Review the retrieved logs injected into history and "
        "compile a detailed, thorough root-cause analysis and mitigation action items roadmap layout.\n"
        "Respond strictly with a JSON object matching this schema:\n"
        '{"analysis": "Your comprehensive engineering report summary and fix list here"}'
    )
    
    response_str = await call_gateway_proxy(prompt, state["messages"])
    data = json.loads(response_str.strip())
    final_report = data.get("analysis", "")
    
    return {"messages": [{"role": "assistant", "content": f"[FINAL SRE REPORT]:\n{final_report}"}]}

# =====================================================================
# 5. CONTROL PLANE ROUTING EDGE
# =====================================================================
def supervisor_router(state: MultiAgentState) -> Literal["log_fetcher_agent_node", "sre_analyst_agent_node", "__end__"]:
    """Reads the next_worker string state mutated by the supervisor node."""
    target = state.get("next_worker")
    
    if target == "log_fetcher_agent":
        return "log_fetcher_agent_node"
    elif target == "sre_analyst_agent":
        return "sre_analyst_agent_node"
    else:
        # If the worker says FINISH, print out the analyst report and terminate
        last_msg = state["messages"][-1]
        last_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
        if "[FINAL SRE REPORT]:" in last_content:
            print(f"\n🤖 {last_content}\n")
        return "__end__"

# =====================================================================
# 6. GRAPH REFACTORING COMPILATION
# =====================================================================
builder = StateGraph(MultiAgentState)

# Register our multi-agent compute nodes
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("log_fetcher_agent_node", log_fetcher_agent_node)
builder.add_node("sre_analyst_agent_node", sre_analyst_agent_node)

# Flow Chart Wiring Configurations
builder.add_edge(START, "supervisor_node")

# Supervisor dynamically determines the worker nodes
builder.add_conditional_edges("supervisor_node", supervisor_router)

# Once a specialised worker completes its subtask, control loops straight back to Supervisor
builder.add_edge("log_fetcher_agent_node", "supervisor_node")
builder.add_edge("sre_analyst_agent_node", "supervisor_node")

app = builder.compile()

# =====================================================================
# 7. EXECUTION PLANE INVOCATION
# =====================================================================
async def main():
    initial_input = {"messages": [{"role": "user", "content": "The auth-service is failing token validation loops. Why?"}]}
    print("👤 [USER]: The auth-service is failing token validation loops. Why?\n")
    await app.ainvoke(initial_input)

if __name__ == "__main__":
    asyncio.run(main())
