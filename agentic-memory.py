import json
import urllib.request
import os
from typing import Literal
import chromadb

# 🌟 IMPORT THE DATABASE PERSISTENCE ENGINE
from langgraph.checkpoint.sqlite import SqliteSaver

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# =====================================================================
# 1. SETUP DATA SUBSYSTEM (ChromaDB Bootstrap)
# =====================================================================
def get_local_embedding(text_to_embed: str) -> list:
    """Computes high-speed semantic vectors and flattens the output for ChromaDB."""
    url = "http://localhost:11434/api/embed"
    payload = {"model": "mxbai-embed-large", "input": text_to_embed}
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        embeddings_list = result['embeddings']
        
        # 🌟 THE FIX: If the model returns a batch matrix (nested list), grab the first vector slice
        if isinstance(embeddings_list, list) and len(embeddings_list) > 0 and isinstance(embeddings_list[0], list):
            return embeddings_list[0]
        
        return embeddings_list

def initialize_vector_store():
    chroma_client = chromadb.Client()
    try: chroma_client.delete_collection(name="memory_knowledge")
    except Exception: pass
    collection = chroma_client.create_collection(name="memory_knowledge")
    with open("knowledge_base.txt", "r") as f: content = f.read()
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    for index, chunk in enumerate(chunks):
        vector = get_local_embedding(chunk)
        collection.add(embeddings=[vector], documents=[chunk], ids=[f"doc_{index}"])
    return collection

GLOBAL_VECTOR_DB = initialize_vector_store()

# =====================================================================
# 2. STATE AND COMPUTE NODES
# =====================================================================
class AgentState(TypedDict):
    messages: list = add_messages
    loop_count: int

def analyzer_node(state: AgentState) -> dict:
    print("🔄 [NODE]: Analyzer evaluating current state graph...")
    url = "http://localhost:11434/api/chat"
    
    system_prompt = {
        "role": "system",
        "content": (
            "You are a production engineering agent. You communicate strictly using structured data. "
            "If you need to query the vector database runbook, respond with a JSON object matching this schema:\n"
            '{"action": "query_vector_knowledge_base", "argument": "your descriptive search query here"}\n\n'
            "If you have gathered the data and are ready to present the final analysis, respond with a JSON object matching this schema:\n"
            '{"action": "final_answer", "argument": "Your complete architectural analysis here"}\n\n'
            "Do not output any normal conversational text outside of these specified JSON definitions."
        )
    }
    
    formatted_messages = []
    for msg in state["messages"]:
        if hasattr(msg, "content"): formatted_messages.append({"role": getattr(msg, "type", "user"), "content": msg.content})
        elif isinstance(msg, dict): formatted_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    payload = {"model": "llama3.1", "messages": [system_prompt] + formatted_messages, "format": "json", "stream": False}
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        return {"messages": [{"role": "assistant", "content": result['message']['content']}]}

def vector_retrieval_node(state: AgentState) -> dict:
    print("⚙️ [NODE]: Triggering live Vector DB query infrastructure...")
    last_msg = state["messages"][-1]
    last_message_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    response_data = json.loads(last_message_content.strip())
    search_query = response_data.get("argument", "")
    
    query_vector = get_local_embedding(search_query)
    results = GLOBAL_VECTOR_DB.query(query_embeddings=[query_vector], n_results=1)
    retrieved_chunk = results['documents']
    
    injection_payload = (
        f"[AUTOMATED VECTOR DB RESPONSE] The database returned this matching runbook page: '{retrieved_chunk}'. "
        f"Analyze this context, compile your definitive root-cause diagnosis, and output your final_answer JSON object."
    )
    return {"messages": [{"role": "user", "content": injection_payload}]}

def validator_node(state: AgentState) -> dict:
    print("🛡️ [NODE]: Guardrail checking final answer quality boundary...")
    url = "http://localhost:11434/api/chat"
    
    last_msg = state["messages"][-1]
    last_message_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    response_data = json.loads(last_message_content.strip())
    draft_answer = response_data.get("argument", "")
    
    validation_prompt = {
        "role": "system",
        "content": (
            "You are a Senior Systems Auditor. Your job is to review a junior engineer's draft answer. "
            "You must ensure the answer mentions technical remediation parameters (like connections, capacity, or databases). "
            "If the answer is comprehensive, respond with exactly: "
            '{"assessment": "PASS", "feedback": "None"}\n'
            "If the answer lacks clear optimization suggestions, respond with:\n"
            '{"assessment": "FAIL", "feedback": "Your detailed instructions on what data the engineer missed"}\n'
            "Respond ONLY using this valid JSON layout."
        )
    }
    
    payload = {
        "model": "llama3.1",
        "messages": [validation_prompt, {"role": "user", "content": f"Review this draft: {draft_answer}"}],
        "format": "json",
        "stream": False
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        current_count = state.get("loop_count", 0)
        return {
            "messages": [{"role": "user", "content": f"[GUARDRAIL ASSESSMENT INTERCEPT]: {result['message']['content']}"}],
            "loop_count": current_count + 1
        }

# =====================================================================
# 3. EDGE ROUTING CONTROL PLANE
# =====================================================================
def router_edge(state: AgentState) -> Literal["vector_retrieval_node", "validator_node", "__end__"]:
    last_msg = state["messages"][-1]
    last_message_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    try:
        response_data = json.loads(last_message_content.strip())
        action = response_data.get("action")
        if action == "query_vector_knowledge_base": return "vector_retrieval_node"
        elif action == "final_answer": return "validator_node"
    except json.JSONDecodeError: return "__end__"
    return "__end__"

def guardrail_edge(state: AgentState) -> Literal["analyzer_node", "__end__"]:
    if state.get("loop_count", 0) >= 2:
        print("\n🚨 [CIRCUIT BREAKER]: Retries exhausted. Exiting to avoid loop, but checking state persistence...")
        return "__end__"

    last_msg = state["messages"][-1]
    last_message_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    json_str = last_message_content.replace("[GUARDRAIL ASSESSMENT INTERCEPT]:", "").strip()
    
    try:
        audit_data = json.loads(json_str)
        if audit_data.get("assessment") == "PASS":
            print("✅ [GUARDRAIL]: Answer passed quality boundary controls successfully.")
            return "__end__"
        else:
            print(f"❌ [GUARDRAIL]: Audit failed. Loop count: {state.get('loop_count')}. Routing back.")
            return "analyzer_node"
    except Exception:
        return "__end__"

# =====================================================================
# 4. COMPILATION AND LOCAL STORAGE PLUG-IN
# =====================================================================
# 🌟 STEP A: Instantiate a local, embedded SQLite database connection
# This automatically boots up an architectural database file named 'agent_memory.db'
memory_db_saver = SqliteSaver.from_conn_string("agent_memory.db")

workflow = StateGraph(AgentState)
workflow.add_node("analyzer_node", analyzer_node)
workflow.add_node("vector_retrieval_node", vector_retrieval_node)
workflow.add_node("validator_node", validator_node)

workflow.add_edge(START, "analyzer_node")
workflow.add_conditional_edges("analyzer_node", router_edge)
workflow.add_edge("vector_retrieval_node", "analyzer_node")
workflow.add_conditional_edges("validator_node", guardrail_edge)

# 🌟 THE CRUCIAL FIX: Compile inside the file-lock connection context manager!
with SqliteSaver.from_conn_string("agent_memory.db") as memory_db_saver:
    app = workflow.compile(checkpointer=memory_db_saver)
    
    print("--- STEP 1: INITIAL COMPUTE RUN ---")
    config_session = {"configurable": {"thread_id": "systems_engineer_session_99"}}
    initial_input = {
        "messages": [{"role": "user", "content": "Our users are hitting token loop timeout validation errors. Why is the service breaking?"}],
        "loop_count": 0
    }
    
    # Process the state graph execution path
    app.invoke(initial_input, config=config_session)
    
    print("\n" + "="*50 + "\n")
    print("--- STEP 2: DISCONNECTED SESSION RESUMPTION ---")
    print("Directly querying state from the local disk file mapping...")
    
    # Read the historical data snapshot back out of the binary database file
    saved_state_snapshot = app.get_state(config_session)
    
    print(f"\n📁 Successfully verified records inside 'agent_memory.db'.")
    print(f"📈 Recovered state 'loop_count' score directly from hard drive layout: {saved_state_snapshot.values.get('loop_count')}")
    
    historical_messages = saved_state_snapshot.values.get("messages", [])
    if historical_messages:
        last_msg = historical_messages[-1]
        last_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
        print(f"📋 Verified Last State Message present in DB: '{last_content[:60]}...'")