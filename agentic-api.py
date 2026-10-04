import json
import logging
import httpx
import chromadb
import uvicorn
from typing import Literal, List
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

# Import LangGraph + Embedded SQLite Checkpoint Persistence Systems
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# =====================================================================
# 1. SETUP INITIAL DATA PLANES (ChromaDB Bootstrap Subsystem)
# =====================================================================
async def get_local_embedding_async(
    client: httpx.AsyncClient, text: str
) -> list[list[float]]:
    url = "http://localhost:11434/api/embed"
    payload = {"model": "mxbai-embed-large", "input": text}
    response = await client.post(url, json=payload, timeout=30.0)
    return response.json()['embeddings']

def initialize_vector_store():
    chroma_client = chromadb.Client()
    try: chroma_client.delete_collection(name="api_knowledge_base")
    except Exception: pass
    collection = chroma_client.create_collection(name="api_knowledge_base")
    with open("knowledge_base.txt", "r") as f: content = f.read()
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    
    import urllib.request
    url = "http://localhost:11434/api/embed"
    for index, chunk in enumerate(chunks):
        payload = {"model": "mxbai-embed-large", "input": chunk}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            collection.add(embeddings=result['embeddings'], documents=[chunk], ids=[f"doc_{index}"])
    return collection

GLOBAL_VECTOR_DB = initialize_vector_store()

# =====================================================================
# 2. STATE AND AGENT COMPUTE NODES (Wired strictly to LiteLLM Port 4000)
# =====================================================================
class AgentState(TypedDict):
    messages: list = add_messages
    next_worker: str
    loop_count: int

async def call_gateway_proxy(system_prompt: str, messages: list) -> str:
    url = "http://localhost:4000/v1/chat/completions"
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if hasattr(msg, "content"):
            role = getattr(msg, "type", "user")
            content = msg.content
        elif isinstance(msg, dict):
            role = msg.get("role", "user")
            content = msg.get("content", "")
        else:
            continue

        role = {"human": "user", "ai": "assistant"}.get(role, role)
        formatted_messages.append({"role": role, "content": content})
        
    payload = {
        "model": "enterprise-brain",
        "messages": formatted_messages,
        "response_format": {"type": "json_object"},
        "stream": False # Static payload targets cache.db allocations
    }
    
    async with httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=5.0)) as client:
        response = await client.post(url, json=payload, headers={"Authorization": "Bearer local-key"})
        response.raise_for_status()
        result = response.json()
        try:
            return result["choices"][0]["message"]["content"]
        except (IndexError, KeyError, TypeError) as exc:
            raise ValueError(
                "Gateway response is missing choices[0].message.content"
            ) from exc

async def supervisor_node(state: AgentState) -> dict:
    prompt = (
        "You are a Technical SRE Supervisor. Review the history context to assign the next step.\n"
        "Your available worker routing choices are:\n"
        "1. 'log_fetcher_agent': Choose this ONLY if logs have not been retrieved from ChromaDB yet.\n"
        "2. 'sre_analyst_agent': Choose this if log details are present in history, but a deep report has not been compiled.\n"
        "3. 'FINISH': Choose this if a comprehensive root-cause analysis report [FINAL SRE REPORT] is already present.\n\n"
        "Respond strictly with a JSON object matching this schema:\n"
        '{"next": "worker_name_or_FINISH"}'
    )
    response_str = await call_gateway_proxy(prompt, state["messages"])
    data = json.loads(response_str.strip())
    return {"next_worker": data.get("next", "FINISH"), "loop_count": state.get("loop_count", 0) + 1}

async def log_fetcher_agent_node(state: AgentState) -> dict:
    prompt = '{"search_query": "search query text"}'
    system_p = f"Generate an optimized search query JSON string matching schema: {prompt} to query runbooks based on history context."
    response_str = await call_gateway_proxy(system_p, state["messages"])
    data = json.loads(response_str.strip())
    search_query = data.get("search_query", "auth-service logs")
    
    async with httpx.AsyncClient() as client:
        query_vector = await get_local_embedding_async(client, search_query)
    results = GLOBAL_VECTOR_DB.query(query_embeddings=query_vector, n_results=1)
    return {"messages": [{"role": "system", "content": f"[SYSTEM RETRIEVAL UPDATE] Logs: '{results['documents']}'."}]}

async def sre_analyst_agent_node(state: AgentState) -> dict:
    prompt = "You are a Principal Engineer. Output a detailed root-cause analysis report labeled with [FINAL SRE REPORT] prefix matching schema: " + '{"analysis": "report text"}'
    response_str = await call_gateway_proxy(prompt, state["messages"])
    data = json.loads(response_str.strip())
    return {"messages": [{"role": "assistant", "content": f"[FINAL SRE REPORT]:\n{data.get('analysis', '')}"}]}

def supervisor_router(state: AgentState) -> Literal["log_fetcher_agent_node", "sre_analyst_agent_node", "__end__"]:
    if state.get("loop_count", 0) >= 4: return "__end__"
    target = state.get("next_worker")
    if target == "log_fetcher_agent": return "log_fetcher_agent_node"
    elif target == "sre_analyst_agent": return "sre_analyst_agent_node"
    return "__end__"

# Assemble Topology Blueprint Objects
builder = StateGraph(AgentState)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("log_fetcher_agent_node", log_fetcher_agent_node)
builder.add_node("sre_analyst_agent_node", sre_analyst_agent_node)
builder.add_edge(START, "supervisor_node")
builder.add_conditional_edges("supervisor_node", supervisor_router)
builder.add_edge("log_fetcher_agent_node", "supervisor_node")
builder.add_edge("sre_analyst_agent_node", "supervisor_node")

# =====================================================================
# 3. FASTAPI REST INTERFACE LAYER (The Network Gateway Server)
# =====================================================================
app = FastAPI(title="Enterprise Agentic SRE Gateway Subsystem API")

# Define strong, typed Pydantic models for request deserialization validation
class UserMessageRequest(BaseModel):
    message: str

class AgentAPIResponse(BaseModel):
    session_id: str
    status: str
    final_output: str

@app.post("/api/chat", response_model=AgentAPIResponse)
async def process_agent_chat_endpoint(
    request: UserMessageRequest, 
    x_session_id: str = Header(default="default_anonymous_thread_uuid") # 🌟 Dynamic header binding!
):
    """Production endpoint executing multi-tenant graph invocations bounded to database session pools."""
    print(f"\n🌐 [HTTP POST]: Incoming request packet captured on thread partition: '{x_session_id}'")
    
    # Establish our structural compilation session context configurations
    config_session = {"configurable": {"thread_id": x_session_id}}
    
    try:
        # Keep checkpoint database operations asynchronous within the request loop.
        async with AsyncSqliteSaver.from_conn_string("api_memory.db") as file_db_checkpointer:
            compiled_graph = builder.compile(checkpointer=file_db_checkpointer)
            
            # Formulate state dictionary parameters mapping
            initial_input = {
                "messages": [{"role": "user", "content": request.message}],
                "loop_count": 0
            }
            
            # Execute the graph workflow asynchronously over the database session token mapping
            final_state_snapshot = await compiled_graph.ainvoke(initial_input, config=config_session)
            
            # Search memory logs backwards to find the terminal SRE analyst report payload string
            final_report_string = "No engineering analysis was generated for this turn."
            for msg in reversed(final_state_snapshot.get("messages", [])):
                content = msg.content if hasattr(msg, "content") else msg.get("content", "")
                if "[FINAL SRE REPORT]:" in content:
                    final_report_string = content.replace("[FINAL SRE REPORT]:", "").strip()
                    break
            
            # Serialize data structures cleanly back down the HTTP response network pipe
            return AgentAPIResponse(
                session_id=x_session_id,
                status="success",
                final_output=final_report_string
            )
            
    except Exception as systems_panic_exception:
        logging.exception(
            "Internal system REST request failed: %s",
            systems_panic_exception,
        )
        raise HTTPException(status_code=500, detail="Internal AI compute engine orchestration error.")

# Bootstraps Uvicorn runtime server loop natively on initialization
if __name__ == "__main__":
    uvicorn.run("agentic-api:app", host="127.0.0.1", port=8000, reload=False)
