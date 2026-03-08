import json
import os

from mcp_router import NeuralMCPRouter

def generate_slm_prompt(user_query, target_schema):
    """Generates the token-efficient instruction for the LLM."""
    return f"""SYSTEM DIRECTIVE:
You are an intelligent agent operating an MCP server.
Execute the user's request using the single tool provided below.

USER REQUEST: "{user_query}"

AUTHORIZED TOOL SCHEMA:
{json.dumps(target_schema, indent=2)}

INSTRUCTION: Output strictly the JSON arguments required to call this tool. Do not explain your reasoning.
"""

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    router = NeuralMCPRouter(
        model_path=os.path.join(base_dir, "models", "best_mcp_router"),
        dataset_path=os.path.join(base_dir, "mcp_routing_dataset.csv"),
    )
    
    # Simulate a user request
    test_query = "Can you find the pull request where we fixed the memory leak in the auth module?"
    print(f"User Prompt: {test_query}\n")
    
    schema, ms = router.route_with_latency(test_query)
    print(f"Routed Tool: {schema.get('name')} (Lat: {ms:.2f}ms)\n")
    
    final_prompt = generate_slm_prompt(test_query, schema)
    print("--- OPTIMIZED SLM PROMPT ---")
    print(final_prompt)