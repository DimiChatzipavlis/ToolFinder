import os
import pandas as pd
import time
import torch
from sentence_transformers import util

from mcp_router import NeuralMCPRouter

def run_zero_shot_eval():
    print("--- INITIATING ZERO-SHOT EVALUATION ---")
    
    # 1. Hardened Path Resolution (Prevents Windows Zip/Nesting Errors)
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "models", "best_mcp_router")
    dataset_v2_path = os.path.join(base_dir, "mcp_routing_dataset_v2.csv")
    
    assert os.path.exists(os.path.join(model_path, "config.json")), f"CRITICAL FAULT: Model not found at {model_path}"
    assert os.path.exists(dataset_v2_path), "CRITICAL FAULT: mcp_routing_dataset_v2.csv not found."

    # 2. Load Model & Unseen Data
    df_v2 = pd.read_csv(dataset_v2_path)
    
    # Extract the 15 NEW tools to form a totally blind corpus
    unseen_corpus_schemas = df_v2['positive_schema'].unique().tolist()
    print(f"Total rows in V2: {len(df_v2)}")
    print(f"Total strictly unseen MCP tools in corpus: {len(unseen_corpus_schemas)}\n")
    
    print(f"Loading 109M Model on {( 'CUDA' if torch.cuda.is_available() else 'CPU')}...")
    router = NeuralMCPRouter(model_path=model_path, dataset_path=dataset_v2_path)

    print("Embedding Unseen Corpus...")
    corpus_embeddings = router.model.encode(unseen_corpus_schemas, convert_to_tensor=True)
    
    # 3. The Zero-Shot Loop
    print("Evaluating Zero-Shot Routing...")
    recall_1, recall_3 = 0, 0
    latencies = []
    
    for _, row in df_v2.iterrows():
        query = row['anchor']
        true_schema = row['positive_schema']
        
        t0 = time.time()
        query_embedding = router.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
        
        # Get Top 3
        top_results = torch.topk(cos_scores, k=min(3, len(unseen_corpus_schemas)))
        top_indices = top_results[1].cpu().tolist()
        
        latencies.append((time.time() - t0) * 1000)
        
        top_1_schema = unseen_corpus_schemas[top_indices[0]]
        top_3_schemas = [unseen_corpus_schemas[i] for i in top_indices]
        
        if true_schema == top_1_schema: recall_1 += 1
        if true_schema in top_3_schemas: recall_3 += 1

    r1_pct = (recall_1 / len(df_v2)) * 100
    r3_pct = (recall_3 / len(df_v2)) * 100
    avg_latency = sum(latencies) / len(latencies)

    print("========================================")
    print("   ZERO-SHOT GENERALIZATION RESULTS     ")
    print("========================================")
    print(f"Recall@1:    {r1_pct:.2f}%")
    print(f"Recall@3:    {r3_pct:.2f}%")
    print(f"Avg Latency: {avg_latency:.2f} ms / query")
    print("========================================")

if __name__ == "__main__":
    run_zero_shot_eval()