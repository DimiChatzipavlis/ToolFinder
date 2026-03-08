import pandas as pd
import torch
import json
import time
from sentence_transformers import SentenceTransformer, util

class NeuralMCPRouter:
    def __init__(self, model_path, dataset_path):
        print("Initializing 109M Parameter Neural Router...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_path, device=self.device)
        
        df = pd.read_csv(dataset_path)
        # Extract the 15 unique schemas
        self.corpus_schemas = df['positive_schema'].unique().tolist()
        self.corpus_embeddings = self.model.encode(self.corpus_schemas, convert_to_tensor=True)
        print("Router Online.\n")

    def route(self, query):
        t0 = time.time()
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        
        # Get exact Top 1
        top_index = torch.topk(cos_scores, k=1)[1].item()
        best_schema_str = self.corpus_schemas[top_index]
        
        latency = (time.time() - t0) * 1000
        return json.loads(best_schema_str), latency

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
    router = NeuralMCPRouter(
        model_path="./models/best_mcp_router", 
        dataset_path="mcp_routing_dataset.csv"
    )
    
    # Simulate a user request
    test_query = "Can you find the pull request where we fixed the memory leak in the auth module?"
    print(f"User Prompt: {test_query}\n")
    
    schema, ms = router.route(test_query)
    print(f"Routed Tool: {schema.get('name')} (Lat: {ms:.2f}ms)\n")
    
    final_prompt = generate_slm_prompt(test_query, schema)
    print("--- OPTIMIZED SLM PROMPT ---")
    print(final_prompt)