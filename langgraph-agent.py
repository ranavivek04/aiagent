import json
import urllib.request
from typing import Literal
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# =====================================================================
# 1. ARCHITECTURAL STATE DEFINITION
# =====================================================================
# In LangGraph, state is explicitly typed and passed between nodes.
# We define an append-only ledger using the 'add_messages' reducer.
class AgentState(TypedDict):
    messages: list = add_messages

# =====================================================================
# 2. ISOLATED COMPUTE NODES (The Workers)
# =====================================================================

def analyzer_node(state: AgentState) -> dict:
    """Node 1: Evaluates current history and generates structured decisions."""
    print("🔄 [NODE]: Analyzer evaluating current state graph...")
    
    url = "http://localhost:11434/api/chat"
    
    system_prompt = {
        "role": "system",
        "content": (
            "You are a production engineering agent. You communicate strictly using structured data. "
            "If you need to fetch system logs to solve a user problem, respond with a JSON object matching this schema:\n"
            '{"action": "fetch_server_logs", "argument": "service_name"}\n\n'
            "If you have gathered the data and are ready to present the final analysis to the user, respond with a JSON object matching this schema:\n"
            '{"action": "final_answer", "argument": "Your complete architectural analysis here"}\n\n'
            "Do not output any normal conversational text outside of these specified JSON definitions."
        )
    }
    
    # 🌟 FIX 1: Normalize incoming messages so Ollama safely receives text dictionaries
    formatted_messages = []
    for msg in state["messages"]:
        if hasattr(msg, "content"):  # If LangGraph converted it to a Message object
            formatted_messages.append({"role": getattr(msg, "type", "user"), "content": msg.content})
        elif isinstance(msg, dict):  # If it's a raw native dict
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
        response_content = result['message']['content']
        
        return {"messages": [{"role": "assistant", "content": response_content}]}


def fetch_logs_node(state: AgentState) -> dict:
    """Node 2: Real systems execution tool that updates the environment."""
    print("⚙️ [NODE]: Natively executing log lookup infrastructure...")
    
    # 🌟 FIX 2: Dynamic attribute checking to extract assistant content safely
    last_msg = state["messages"][-1]
    last_message_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    
    response_data = json.loads(last_message_content.strip())
    argument = response_data.get("argument", "unknown")
    
    clean_argument = argument.replace(".", "-").replace("_", "-")
    
    mock_db = {
        "auth-service": "[2026-09-12 08:12:00] ERROR: Token validation failed. Database connection timeout.",
        "payment-gateway": "[2026-09-12 08:14:15] INFO: Transaction successful for user_9934.",
        "inventory-manager": "[2026-09-12 08:15:00] WARN: Low stock alert on item_id_554."
    }
    
    tool_output = mock_db.get(clean_argument, f"No logs found for service: {clean_argument}")
    print(f"📥 [TOOL RESULT]: Found matching context block for '{clean_argument}'")
    
    injection_payload = (
        f"[AUTOMATED TOOL RESPONSE] The system successfully executed fetch_server_logs for '{clean_argument}'. "
        f"The logs retrieved are: '{tool_output}'. "
        f"Now, evaluate this data and output your final_answer JSON object. Do not request the tool again."
    )
    
    return {"messages": [{"role": "user", "content": injection_payload}]}

# =====================================================================
# 3. THE ROUTING CONTROL PLANE (The Conditional Edge)
# =====================================================================

def router_edge(state: AgentState) -> Literal["fetch_logs_node", "__end__"]:
    """Evaluates data from the preceding node and routes traffic accordingly."""
    # 🌟 FIX 3: Dynamic attribute checking to read routing state safely
    last_msg = state["messages"][-1]
    last_message_content = last_msg.content if hasattr(last_msg, "content") else last_msg.get("content", "")
    
    try:
        response_data = json.loads(last_message_content.strip())
        action = response_data.get("action")
        
        if action == "fetch_server_logs":
            return "fetch_logs_node"
            
        elif action == "final_answer":
            print(f"\n🤖 [AGENT FINAL ANSWER]:\n{response_data.get('argument')}\n")
            return "__end__"
            
    except json.JSONDecodeError:
        print(f"⚠️ State Boundary Error: Unparsable content generated: {last_message_content}")
        return "__end__"
        
    return "__end__"

# =====================================================================
# 4. COMPILING THE DIRECTED STATE GRAPH
# =====================================================================

# Define a workflow initialized with our explicit State blueprint
workflow = StateGraph(AgentState)

# Register our compute nodes
workflow.add_node("analyzer_node", analyzer_node)
workflow.add_node("fetch_logs_node", fetch_logs_node)

# Set the Entry Point configuration of the graph
workflow.add_edge(START, "analyzer_node")

# Establish a Conditional Routing Edge connecting nodes dynamically
workflow.add_conditional_edges(
    "analyzer_node",
    router_edge
)

# Connect the tool output back into the evaluation loop
workflow.add_edge("fetch_logs_node", "analyzer_node")

# Compile the graph into an executable state machine runtime interface
app = workflow.compile()

# =====================================================================
# 5. EXECUTION PLAN INFRASTRUCTURE
# =====================================================================
if __name__ == "__main__":
    initial_input = {"messages": [{"role": "user", "content": "Can you find out why the auth-service is failing?"}]}
    
    print("👤 [USER]: Can you find out why the auth-service is failing?\n")
    
    # Trigger the asynchronous run cycle across the graph
    app.invoke(initial_input)
