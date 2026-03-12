import os
import json
import time
import hashlib

import faiss
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


class NeuralMCPRouter:
    @staticmethod
    def canonicalize_schema(schema_str):
        return json.dumps(
            json.loads(schema_str),
            sort_keys=True,
            separators=(",", ":"),
        )

    def _minify_schema_for_embedding(self, schema_dict):
        minified = json.loads(json.dumps(schema_dict))
        input_properties = minified.get("inputSchema", {}).get("properties", {})
        for property_schema in input_properties.values():
            if isinstance(property_schema, dict):
                property_schema.pop("description", None)
        return json.dumps(minified, sort_keys=True, separators=(",", ":"))

    def __init__(self, model_path=None, dataset_path=None):
        if dataset_path is None:
            raise ValueError("dataset_path is required")

        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, "models", "best_mcp_router")

        self.model_path = os.fspath(model_path)
        self.dataset_path = os.fspath(dataset_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.faiss_cache_path = None
        self.cache_meta_path = None

        self.model = SentenceTransformer(self.model_path, device=self.device)

        df = pd.read_csv(self.dataset_path)
        raw_schemas = df["positive_schema"].unique().tolist()
        self.corpus_schemas = [self.canonicalize_schema(s) for s in raw_schemas]
        self.embedding_corpus = [
            self._minify_schema_for_embedding(json.loads(schema))
            for schema in self.corpus_schemas
        ]

        if os.path.isdir(self.model_path):
            dataset_name = os.path.basename(self.dataset_path)
            cache_name = dataset_name.replace('.csv', '_faiss.index')
            meta_name = dataset_name.replace('.csv', '_faiss_meta.json')
            self.faiss_cache_path = os.path.join(self.model_path, cache_name)
            self.cache_meta_path = os.path.join(self.model_path, meta_name)

        corpus_signature = hashlib.sha256(
            "\n".join(self.corpus_schemas).encode("utf-8")
        ).hexdigest()

        can_load_cache = False
        if self.faiss_cache_path and self.cache_meta_path and os.path.exists(self.faiss_cache_path) and os.path.exists(self.cache_meta_path):
            try:
                with open(self.cache_meta_path, "r", encoding="utf-8") as meta_file:
                    cache_meta = json.load(meta_file)
                can_load_cache = (
                    cache_meta.get("dataset_path") == os.path.abspath(self.dataset_path)
                    and cache_meta.get("corpus_signature") == corpus_signature
                    and cache_meta.get("schema_count") == len(self.corpus_schemas)
                )
            except (OSError, json.JSONDecodeError):
                can_load_cache = False

        if can_load_cache:
            self.faiss_index = faiss.read_index(self.faiss_cache_path)
        else:
            with torch.inference_mode():
                embeddings = self.model.encode(
                    self.embedding_corpus,
                    convert_to_numpy=True,
                )
            embeddings = np.asarray(embeddings, dtype=np.float32)
            faiss.normalize_L2(embeddings)
            self.faiss_index = faiss.IndexFlatIP(embeddings.shape[1])
            self.faiss_index.add(embeddings)
            if self.faiss_cache_path and self.cache_meta_path:
                faiss.write_index(self.faiss_index, self.faiss_cache_path)
                with open(self.cache_meta_path, "w", encoding="utf-8") as meta_file:
                    json.dump(
                        {
                            "dataset_path": os.path.abspath(self.dataset_path),
                            "corpus_signature": corpus_signature,
                            "schema_count": len(self.corpus_schemas),
                            "cache_type": "faiss_index_flat_ip",
                        },
                        meta_file,
                    )

    def route_top_k(self, query, k=3):
        if k < 1:
            raise ValueError("k must be at least 1")

        if self.device == "cuda":
            torch.cuda.synchronize()
        with torch.inference_mode():
            query_embedding = self.model.encode([query], convert_to_numpy=True)
        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        faiss.normalize_L2(query_embedding)
        _scores, indices = self.faiss_index.search(query_embedding, k=min(k, len(self.corpus_schemas)))
        if self.device == "cuda":
            torch.cuda.synchronize()
        return [self.corpus_schemas[index] for index in indices[0] if index >= 0]

    def route(self, query):
        return json.loads(self.route_top_k(query, k=1)[0])

    def route_with_latency(self, query):
        if self.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        schema = self.route(query)
        if self.device == "cuda":
            torch.cuda.synchronize()
        return schema, (time.time() - t0) * 1000