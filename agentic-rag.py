import json
import urllib.request
from typing import Literal
import chromadb
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# =====================================================================
# 1. SUBSYSTEM A: THE COMPACT DATA PLANE (Vector DB Engine)
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

def initialize_production_vector_store():
    """Builds and fills our vector index layer dynamically from the text file."""
    chroma_client = chromadb.Client()
    try:
        chroma_client.delete_collection(name="production_knowledge")
    except Exception:
        pass
    collection = chroma_client.create_collection(name="production_knowledge")
    
    with open("knowledge_base.txt", "r") as f:
        content = f.read()
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    
    for index, chunk in enumerate(chunks):
        vector = get_local_embedding(chunk)
        collection.add(embeddings=[vector], documents=[chunk], ids=[f"doc_{index}"])
    
    return collection

# Bootstrap the local database collection engine instance globally
GLOBAL_VECTOR_DB = initialize_production_vector_store()

# =====================================================================
# 2. SUBSYSTEM B: THE STATE LEDGER MAP
# =====================================================================
class AgentState(TypedDict):
    messages: list = add_messages

# =====================================================================
# 3. SUBSYSTEM C: GRAPH CORE COMPUTE NODES
# =====================================================================
def analyzer_node(state: AgentState) -> dict:
    """Node 1: The Agentic Brain. Evaluates execution paths via strict JSON format."""
    print("🔄 [NODE]: Analyzer evaluating current state graph...")
    url = "http://localhost:11434/api/chat"
    
    system_prompt = {
        "role": "system",
        "content": (
            "You are a production engineering agent. You communicate strictly using structured data. "
            "If you need to query the vector database runbook to solve a user problem, respond with a JSON object matching this schema:\n"
            '{"action": "query_vector_knowledge_base", "argument": "your descriptive search query here"}\n\n'
            "If you have gathered the data and are ready to present the final analysis to the user, respond with a JSON object matching this schema:\n"
            '{"action": "final_answer", "argument": "Your complete architectural analysis here"}\n\n'
            "Do not output any normal conversational text outside of these specified JSON definitions."
        )
    }
    
    formatted_messages = []
    for msg in state["messages"]:
        if hasattr(msg, "content"):
            formatted_messages.append({"role": getattr(msg, "type", "user"), "content": msg.content})
        elif isinstance(msg, dict):
            formatted_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})

    payload = {
        "model": "llama3.1",
        "messages": [system_prompt] + formatted_messages,
        "format": "json",
        "stream": False
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        return {"messages": [{"role": "assistant", "content": result['message']['content']}]}


def vector_retrieval_node(state: AgentState) -> dict:
    """Node 2: The Agent Data Tool. Connects the agent directly to ChromaDB."""
    print("⚙️ [NODE]: Triggering live Vector DB query infrastructure...")
    
    last_msg = state["messages"][-1]
    last_message_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    
    response_data = json.loads(last_message_content.strip())
    search_query = response_data.get("argument", "")
    
    print(f"🔎 [VECTOR SEARCH PARAMETER]: '{search_query}'")
    
    # Run dynamic embedding parsing on the agent's chosen query string
    query_vector = get_local_embedding(search_query)
    results = GLOBAL_VECTOR_DB.query(query_embeddings=[query_vector], n_results=1)
    
    retrieved_chunk = results['documents'][0][0]
    print("📥 [DATA FOUND]: Relevant architecture context extracted successfully.")
    
    injection_payload = (
        f"[AUTOMATED VECTOR DB RESPONSE] The database returned this matching runbook page: '{retrieved_chunk}'. "
        f"Analyze this context, compile your definitive root-cause diagnosis, and output your final_answer JSON object."
    )
    
    return {"messages": [{"role": "user", "content": injection_payload}]}

# =====================================================================
# 4. SUBSYSTEM D: THE GRAPH ROUTER PLANE (Edges)
# =====================================================================
def router_edge(state: AgentState) -> Literal["vector_retrieval_node", "__end__"]:
    last_msg = state["messages"][-1]
    last_message_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    
    try:
        response_data = json.loads(last_message_content.strip())
        action = response_data.get("action")
        
        if action == "query_vector_knowledge_base":
            return "vector_retrieval_node"
        elif action == "final_answer":
            print(f"\n🤖 [AGENT FINAL ANSWER]:\n{response_data.get('argument')}\n")
            return "__end__"
    except json.JSONDecodeError:
        print("⚠️ State Boundary Error: Unparsable content generated.")
        return "__end__"
    return "__end__"

# =====================================================================
# 5. WORKFLOW COMPILATION AND ORCHESTRATION
# =====================================================================
workflow = StateGraph(AgentState)
workflow.add_node("analyzer_node", analyzer_node)
workflow.add_node("vector_retrieval_node", vector_retrieval_node)

workflow.add_edge(START, "analyzer_node")
workflow.add_conditional_edges("analyzer_node", router_edge)
workflow.add_edge("vector_retrieval_node", "analyzer_node")

app = workflow.compile()

if __name__ == "__main__":
    # Note how the initial question implies a search but doesn't name the database type
    initial_input = {"messages": [{"role": "user", "content": "Our users are hitting token loop timeout validation errors. Why is the service breaking?"}]}
    print("👤 [USER]: Our users are hitting token loop timeout validation errors. Why is the service breaking?\n")
    app.invoke(initial_input)
