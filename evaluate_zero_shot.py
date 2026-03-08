import os
import json
import pandas as pd
import time
import torch
import faiss
import numpy as np

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
    unseen_corpus_schemas = [
        NeuralMCPRouter.canonicalize_schema(schema)
        for schema in df_v2['positive_schema'].unique().tolist()
    ]
    print(f"Total rows in V2: {len(df_v2)}")
    print(f"Total strictly unseen MCP tools in corpus: {len(unseen_corpus_schemas)}\n")

    print(f"Loading 109M Model on {( 'CUDA' if torch.cuda.is_available() else 'CPU')}...")
    router = NeuralMCPRouter(model_path=model_path, dataset_path=dataset_v2_path)
    minified_corpus_schemas = [
        router._minify_schema_for_embedding(json.loads(schema))
        for schema in unseen_corpus_schemas
    ]

    print("Embedding Unseen Corpus...")
    with torch.inference_mode():
        corpus_embeddings = router.model.encode(minified_corpus_schemas, convert_to_numpy=True)
    corpus_embeddings = np.asarray(corpus_embeddings, dtype=np.float32)
    faiss.normalize_L2(corpus_embeddings)
    faiss_index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
    faiss_index.add(corpus_embeddings)
    
    # 3. The Zero-Shot Loop
    print("Evaluating Zero-Shot Routing...")
    recall_1, recall_3 = 0, 0
    latencies = []
    
    for _, row in df_v2.iterrows():
        query = row['anchor']
        true_schema = NeuralMCPRouter.canonicalize_schema(row['positive_schema'])
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        with torch.inference_mode():
            query_embedding = router.model.encode([query], convert_to_numpy=True)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        _scores, top_indices = faiss_index.search(query_embedding, k=min(3, len(unseen_corpus_schemas)))
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        latencies.append((time.time() - t0) * 1000)

        ranked_indices = top_indices[0].tolist()
        top_1_schema = unseen_corpus_schemas[ranked_indices[0]]
        top_3_schemas = [unseen_corpus_schemas[i] for i in ranked_indices]
        
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