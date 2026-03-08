import os
import json
import urllib.request

from mcp_router import NeuralMCPRouter

def generate_prompt(user_query, target_schema):
    return f"""SYSTEM DIRECTIVE:
You are an intelligent agent operating an MCP server. Execute the user's request using the single tool provided below.

USER REQUEST: "{user_query}"

AUTHORIZED TOOL SCHEMA:
{json.dumps(target_schema, indent=2)}

INSTRUCTION: Output strictly the JSON arguments required to call this tool. Do not explain your reasoning."""

def call_local_llm(prompt):
    """Calls local Ollama via its native REST API instead of a brittle CLI pipe."""
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.2",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        req = urllib.request.Request(
            url, 
            data=json.dumps(payload).encode('utf-8'), 
            headers={'Content-Type': 'application/json'}
        )
        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode('utf-8'))
            return result.get("response", "").strip()
    except urllib.error.URLError:
        return "API Error: Ollama is not running. Please start the Ollama application."
    except Exception as e:
        return f"Unexpected Error: {str(e)}"

if __name__ == "__main__":
    print("Initializing Local Copilot Backend (109M Router)...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    router = NeuralMCPRouter(
        model_path=os.path.join(base_dir, "models", "best_mcp_router"),
        dataset_path=os.path.join(base_dir, "mcp_routing_dataset.csv"),
    )
    print("Backend Online. Type 'exit' to quit.\n")
    
    while True:
        user_input = input("\n👤 You: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        print("⚙️  Routing query...")
        schema, latency = router.route_with_latency(user_input)
        print(f"✔️  Tool Selected: {schema.get('name')} (in {latency:.2f}ms)")
        
        print("🧠 Generating execution parameters via Llama 3.2...")
        final_prompt = generate_prompt(user_input, schema)
        
        llm_response = call_local_llm(final_prompt)
        
        print("\n🤖 Local Agent Output:")
        print(llm_response)