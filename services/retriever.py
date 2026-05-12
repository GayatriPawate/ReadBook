from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from clients.sarvam_client import BaseSarvamClient


@dataclass
class Retriever:
    chunks: List[Dict]
    vectors: np.ndarray
    backend: str
    client: BaseSarvamClient

    @classmethod
    def from_chunks(cls, chunks: List[Dict], client: BaseSarvamClient) -> "Retriever":
        texts = [chunk["text"] for chunk in chunks]
        vectors = np.array(client.embed(texts))
        return cls(
            chunks=chunks,
            vectors=vectors,
            backend="sarvam_or_fallback",
            client=client,
        )

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        query_vector = self._embed_query(query)
        scores = self._cosine_similarity(query_vector, self.vectors)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results: List[Dict] = []
        for index in top_indices:
            item = dict(self.chunks[index])
            item["score"] = float(scores[index])
            results.append(item)
        return results

    def _embed_query(self, query: str) -> np.ndarray:
        try:
            query_vector = np.array(self.client.embed([query])[0])
            if query_vector.shape[0] == self.vectors.shape[1]:
                return query_vector
        except Exception:
            pass

        vectorizer = TfidfVectorizer(max_features=self.vectors.shape[1] or 256)
        corpus = [chunk["text"] for chunk in self.chunks] + [query]
        matrix = vectorizer.fit_transform(corpus).toarray()
        self.vectors = matrix[:-1]
        return matrix[-1]

    @staticmethod
    def _cosine_similarity(query_vector: np.ndarray, doc_vectors: np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_vector) or 1.0
        doc_norms = np.linalg.norm(doc_vectors, axis=1)
        doc_norms[doc_norms == 0] = 1.0
        return np.dot(doc_vectors, query_vector) / (doc_norms * query_norm)
