import json
import asyncio  
import httpx    
import chromadb
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
import urllib.request 

# =====================================================================
# 1. DATA PLANE CONFIGURATION (ChromaDB Core Setup)
# =====================================================================
async def get_local_embedding_async(client: httpx.AsyncClient, text: str) -> list:
    """Non-blocking asynchronous vector embedding fetch."""
    url = "http://localhost:11434/api/embed"
    payload = {"model": "mxbai-embed-large", "input": text}
    response = await client.post(url, json=payload, timeout=30.0)
    result = response.json()
    embeddings = result['embeddings']
    return embeddings[0] if embeddings and isinstance(embeddings[0], list) else embeddings

def initialize_vector_store():
    chroma_client = chromadb.Client()
    try: chroma_client.delete_collection(name="stream_knowledge")
    except Exception: pass
    collection = chroma_client.create_collection(name="stream_knowledge")
    with open("knowledge_base.txt", "r") as f: content = f.read()
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    
    print("💾 Generating embeddings and loading vector space...")
    url = "http://localhost:11434/api/embed"
    for index, chunk in enumerate(chunks):
        payload = {"model": "mxbai-embed-large", "input": chunk}
        req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            embeddings_list = result['embeddings']

            # Ollama returns a batch of vectors; Chroma expects one flat vector per record.
            vector = embeddings_list[0] if embeddings_list and isinstance(embeddings_list[0], list) else embeddings_list
            collection.add(
                embeddings=[vector], 
                documents=[chunk], 
                ids=[f"doc_{index}"]
            )
    return collection

GLOBAL_VECTOR_DB = initialize_vector_store()

# =====================================================================
# 2. STATE AND ASYNCHRONOUS COMPUTE NODES
# =====================================================================
class AgentState(TypedDict):
    messages: list = add_messages

async def analyzer_node(state: AgentState) -> dict:
    print("\n🔄 [NODE]: Analyzer evaluating current state graph...")
    print("🌊 [STREAM STARTING]: ", end="", flush=True)
    
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

    payload = {
        "model": "llama3.1", 
        "messages": [system_prompt] + formatted_messages, 
        "format": "json", 
        "stream": True 
    }
    
    complete_response_string = ""
    
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload, timeout=60.0) as stream:
            async for line in stream.aiter_lines():
                if line:
                    chunk_json = json.loads(line)
                    token = chunk_json.get("message", {}).get("content", "")
                    complete_response_string += token
                    
                    # 🌟 THE UX GAIN: Stream characters directly to the console immediately!
                    print(token, end="", flush=True)
                    
    print(" 🌊 [STREAM COMPLETED]")
    return {"messages": [{"role": "assistant", "content": complete_response_string}]}


async def vector_retrieval_node(state: AgentState) -> dict:
    print("⚙️ [NODE]: Triggering live Vector DB query infrastructure...")
    last_msg = state["messages"][-1]
    last_message_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    response_data = json.loads(last_message_content.strip())
    search_query = response_data.get("argument", "")
    
    print(f"🔎 [VECTOR SEARCH PARAMETER]: '{search_query}'")
    
    async with httpx.AsyncClient() as client:
        query_vector = await get_local_embedding_async(client, search_query)
        
    results = GLOBAL_VECTOR_DB.query(query_embeddings=[query_vector], n_results=1)
    retrieved_chunk = results['documents']
    
    injection_payload = (
        f"[AUTOMATED VECTOR DB RESPONSE] The database returned this matching runbook page: '{retrieved_chunk}'. "
        f"Analyze this context, compile your definitive root-cause diagnosis, and output your final_answer JSON object."
    )
    return {"messages": [{"role": "user", "content": injection_payload}]}

# =====================================================================
# 3. EDGE ROUTING PLANE (Control Flow)
# =====================================================================
def router_edge(state: AgentState) -> Literal["vector_retrieval_node", "__end__"]:
    last_msg = state["messages"][-1]
    last_message_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    try:
        response_data = json.loads(last_message_content.strip())
        action = response_data.get("action")
        if action == "query_vector_knowledge_base": return "vector_retrieval_node"
        elif action == "final_answer":
            print(f"\n🤖 [AGENT FINAL ANSWER VIA STREAM]:\n{response_data.get('argument')}\n")
            return "__end__"
    except json.JSONDecodeError:
        return "__end__"
    return "__end__"

# =====================================================================
# 4. COMPILATION AND GRAPH ORCHESTRATION
# =====================================================================
workflow = StateGraph(AgentState)
workflow.add_node("analyzer_node", analyzer_node)
workflow.add_node("vector_retrieval_node", vector_retrieval_node)

workflow.add_edge(START, "analyzer_node")
workflow.add_conditional_edges("analyzer_node", router_edge)
workflow.add_edge("vector_retrieval_node", "analyzer_node")

app = workflow.compile()

# =====================================================================
# 5. ASYNCHRONOUS ENGINE INVOCATION
# =====================================================================
async def main():
    initial_input = {"messages": [{"role": "user", "content": "Our users are hitting token loop timeout validation errors. Why is the service breaking?"}]}
    print("👤 [USER]: Our users are hitting token loop timeout validation errors. Why is the service breaking?")
    await app.ainvoke(initial_input)

if __name__ == "__main__":
    asyncio.run(main())
