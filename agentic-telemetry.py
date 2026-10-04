import json
import asyncio
import httpx
from typing import Literal

# =====================================================================
# 🌟 OPENTELEMETRY TRACING AGENTIC INSTRUMENTATION PLANE
# =====================================================================
from openinference.instrumentation.openai import OpenAIInstrumentor
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

print("🛡️ Bootstrapping Local OpenTelemetry Agentic Instrumentation Plane...")
# Direct outgoing trace payloads directly to our running Phoenix server at port 6006
phoenix_endpoint = "http://localhost:6006/v1/traces"
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=phoenix_endpoint)))
trace.set_tracer_provider(trace_provider)

# Auto-instrument all outgoing OpenAI-protocol HTTP requests across your system hooks
OpenAIInstrumentor().instrument()

# =====================================================================
# GRAPH WORKFLOW AND MULTI-AGENT COMPUTE (Wired straight to port 4000)
# =====================================================================
from langgraph.graph import StateGraph, START
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

class AgentState(TypedDict):
    messages: list = add_messages

async def call_gateway_with_telemetry(system_prompt: str, messages: list) -> str:
    url = "http://localhost:4000/v1/chat/completions"
    
    formatted_messages = [{"role": "system", "content": system_prompt}]
    for msg in messages:
        if hasattr(msg, "content"): formatted_messages.append({"role": getattr(msg, "type", "user"), "content": msg.content})
        elif isinstance(msg, dict): formatted_messages.append({"role": msg.get("role", "user"), "content": msg.get("content", "")})
        
    payload = {
        "model": "enterprise-brain",
        "messages": formatted_messages,
        "response_format": {"type": "json_object"},
        "stream": False # Static mode active to capture complete request logs cleanly
    }
    
    # Open a non-blocking asynchronous HTTP network request
    async with httpx.AsyncClient(timeout=None) as client:
        with trace.get_tracer(__name__).start_as_current_span("gateway.chat.completions") as span:
            span.set_attribute("server.address", "localhost")
            span.set_attribute("server.port", 4000)
            span.set_attribute("gen_ai.request.model", payload["model"])
            response = await client.post(url, json=payload, headers={"Authorization": "Bearer local-key"})
            response.raise_for_status()
            result_data = response.json()
            return result_data['choices'][0]['message']['content']

async def observable_node(state: AgentState) -> dict:
    print("\n🔄 [NODE]: Running telemetry instrumented logic loops...")
    prompt = "You are a systems health check tool. Respond with a JSON object containing a key named 'system' set to 'online'."
    
    response_str = await call_gateway_with_telemetry(prompt, state["messages"])
    print(f"📥 [RESPONSE CAPTURED FROM GATEWAY]: {response_str.strip()}")
    return {"messages": [{"role": "assistant", "content": response_str}]}

# Compile standard workflow
workflow = StateGraph(AgentState)
workflow.add_node("observable_node", observable_node)
workflow.add_edge(START, "observable_node")
app = workflow.compile()

async def main():
    print("🚀 Triggering Instrumented Agent Run...")
    with trace.get_tracer(__name__).start_as_current_span("agent.run") as span:
        span.set_attribute("gen_ai.operation.name", "invoke_agent")
        await app.ainvoke({"messages": [{"role": "user", "content": "Execute full diagnostics tracing parameters."}]})

if __name__ == "__main__":
    asyncio.run(main())
  # 🌟 THE LIFECYCLE FIX: FORCE EXPLICIT FLUSH BEFORE SCRIPT EXIT
    # This prevents the script from closing until the background OpenTelemetry 
    # worker thread completes pushing its JSON payloads over port 6006!
    print("⏳ Flushing buffer entries to local telemetry sink...")
    trace_provider.shutdown()
    print("✅ System logs committed. Exiting application process context.")