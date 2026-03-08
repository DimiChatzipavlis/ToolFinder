import os
import json

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util


class NeuralMCPRouter:
    def __init__(self, model_path, dataset_path):
        self.model_path = os.fspath(model_path)
        self.dataset_path = os.fspath(dataset_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SentenceTransformer(self.model_path, device=self.device)

        df = pd.read_csv(self.dataset_path)
        self.corpus_schemas = df["positive_schema"].unique().tolist()
        self.corpus_embeddings = self.model.encode(
            self.corpus_schemas,
            convert_to_tensor=True,
        )

    def route(self, query):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_index = torch.topk(cos_scores, k=1)[1].item()
        return json.loads(self.corpus_schemas[top_index])

    def route_with_latency(self, query):
        if self.device == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            schema = self.route(query)
            end_event.record()
            torch.cuda.synchronize()
            return schema, start_event.elapsed_time(end_event)

        import time

        t0 = time.time()
        schema = self.route(query)
        return schema, (time.time() - t0) * 1000