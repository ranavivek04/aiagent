import json
import urllib.request

# 1. CENTRAL LEDGER STATE
conversation_history = []

# 2. DETERMINISTIC CORE SYSTEM TOOLS
def fetch_server_logs(service_name: str) -> str:
    mock_db = {
        "auth-service": "[2026-09-12 08:12:00] ERROR: Token validation failed. Database connection timeout.",
        "payment-gateway": "[2026-09-12 08:14:15] INFO: Transaction successful for user_9934.",
        "inventory-manager": "[2026-09-12 08:15:00] WARN: Low stock alert on item_id_554."
    }
    print(f"⚙️ [SYSTEM TOOL]: Natively executing fetch_server_logs for: '{service_name}'")
    return mock_db.get(service_name, f"No logs found for service: {service_name}")

# 3. CONSTRAINED INFRASTRUCTURE ROUTER (Hits local microservice)
def query_local_llm(messages):
    url = "http://localhost:11434/api/chat"
    
    # Strict Schema Prompts forcing deterministic routing parameters
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
    
    payload = {
        "model": "llama3.1",  # 🌟 Wired natively to your newly pulled model
        "messages": [system_prompt] + messages,
        "format": "json",     # 🌟 Enforces Ollama's strict token grammar to avoid malformed output
        "stream": False
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode('utf-8'), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as response:
        result = json.loads(response.read().decode('utf-8'))
        return result['message']['content']

# 4. DETERMINISTIC STATE MACHINE LOOP
def run_agent(user_goal: str):
    print(f"👤 [USER]: {user_goal}\n")
    conversation_history.append({"role": "user", "content": user_goal})
    
    max_loops = 5
    loop_count = 0
    
    while loop_count < max_loops:
        loop_count += 1
        print(f"🔄 Loop Iteration {loop_count}: Evaluating next action graph...")
        
        raw_llm_response = query_local_llm(conversation_history)
        
        # Log response string directly to ledger history for consistency
        conversation_history.append({"role": "assistant", "content": raw_llm_response})
        
        try:
            # Parse the JSON layout directly
            response_data = json.loads(raw_llm_response.strip())
            action = response_data.get("action")
            argument = response_data.get("argument")
            
                        # Scenario A: The Model requests execution of an internal system tool
            if action == "fetch_server_logs":
                clean_argument = argument.replace(".", "-").replace("_", "-")
                
                tool_output = fetch_server_logs(clean_argument)
                
                # 🌟 THE FIX: Explicitly label this as system/tool injection so the model knows it already ran!
                conversation_history.append({
                    "role": "user", 
                    "content": (
                        f"[AUTOMATED TOOL RESPONSE] The system successfully executed fetch_server_logs for '{clean_argument}'. "
                        f"The logs retrieved are: '{tool_output}'. "
                        f"Now, evaluate this data and output your final_answer JSON object. Do not request the tool again."
                    )
                })
                print(f"📥 [TOOL RESULT ADDED TO LEDGER HISTORY]\n")
                continue
                
            # Scenario B: The Model routes execution out of the graph with the final answer
            elif action == "final_answer":
                print(f"\n🤖 [AGENT FINAL ANSWER]:\n{argument}")
                break
                
            else:
                print(f"⚠️ Encountered unknown state routing parameter: {action}")
                break
                
        except json.JSONDecodeError:
            print(f"⚠️ Panic Boundary: Server returned invalid JSON format structural payload: {raw_llm_response}")
            break
            
    if loop_count >= max_loops:
        print("⚠️ Fail-Safe Intercept: System stopped to prevent infinite agent execution loop.")

# 🏁 START AGENT OPERATION
run_agent("Can you find out why the auth-service is failing?")
