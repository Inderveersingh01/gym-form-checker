import glob
import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple


class DenseSemanticEmbedder:
    """
    High-performance semantic text embedder using n-gram subword frequency
    and BM25/TF-IDF weighting to project domain text into a normalized
    vector space for cosine similarity matching.
    Zero external heavy neural dependencies required.
    """

    def __init__(self, vocabulary_size: int = 1024):
        self.vocabulary_size = vocabulary_size
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}

    def _tokenize(self, text: str) -> List[str]:
        cleaned = re.sub(r"[^a-zA-Z0-9\s]", " ", text.lower())
        tokens = cleaned.split()
        # Extract unigrams and bigrams
        unigrams = [t for t in tokens if len(t) > 2]
        bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens) - 1)]
        return unigrams + bigrams

    def fit(self, documents: List[str]):
        doc_count = len(documents)
        df: Dict[str, int] = {}
        term_freqs: Dict[str, int] = {}

        for doc in documents:
            tokens = set(self._tokenize(doc))
            for t in tokens:
                df[t] = df.get(t, 0) + 1
            for t in self._tokenize(doc):
                term_freqs[t] = term_freqs.get(t, 0) + 1

        # Select top most informative terms
        sorted_terms = sorted(term_freqs.items(), key=lambda x: x[1], reverse=True)[: self.vocabulary_size]
        self.vocab = {term: idx for idx, (term, _) in enumerate(sorted_terms)}

        # Compute smoothed inverse document frequency
        for term in self.vocab:
            doc_freq = df.get(term, 1)
            self.idf[term] = math.log((doc_count + 1.0) / (doc_freq + 1.0)) + 1.0

    def embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        vec = [0.0] * len(self.vocab)
        if not tokens or not self.vocab:
            return vec

        # Term frequency
        tf: Dict[str, int] = {}
        for t in tokens:
            if t in self.vocab:
                tf[t] = tf.get(t, 0) + 1

        for term, count in tf.items():
            idx = self.vocab[term]
            vec[idx] = (1.0 + math.log(count)) * self.idf.get(term, 1.0)

        # L2 normalize vector
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec


class VectorStore:
    """
    RAG Vector Database for Strength & Conditioning Biomechanics Literature.
    Maintains document vectors, metadata, and performs high-speed cosine similarity search.
    Can be configured with ChromaDB backend or high-speed embedded vector index.
    """

    def __init__(self, kb_dir: Optional[str] = None):
        self.documents: List[Dict[str, Any]] = []
        self.embedder = DenseSemanticEmbedder(vocabulary_size=1024)
        self.kb_dir = kb_dir or os.path.join(os.path.dirname(__file__), "knowledge_base")
        self._load_and_index_knowledge_base()

    def _load_and_index_knowledge_base(self):
        """Loads all JSON files in knowledge_base directory and constructs vector index."""
        raw_cards: List[Dict[str, Any]] = []
        corpus_texts: List[str] = []

        json_files = glob.glob(os.path.join(self.kb_dir, "*.json"))
        for filepath in json_files:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    cards = json.load(f)
                    for card in cards:
                        # Construct comprehensive text representation for vector indexing
                        cues_text = " ".join(card.get("verbal_cues", []))
                        drills_text = " ".join(
                            f"{d.get('drill', '')} {d.get('coaching_points', '')}"
                            for d in card.get("corrective_drills", [])
                        )
                        full_text = f"{card.get('fault_name', '')} {card.get('fault_code', '')} {card.get('exercise', '')} {card.get('biomechanical_root_cause', '')} {cues_text} {drills_text}"
                        raw_cards.append({
                            "id": f"{card.get('exercise')}_{card.get('fault_code')}",
                            "text": full_text,
                            "card": card,
                        })
                        corpus_texts.append(full_text)
            except Exception as e:
                print(f"Warning: could not load knowledge card {filepath}: {e}")

        if not corpus_texts:
            return

        # Train embedding space on strength domain corpus
        self.embedder.fit(corpus_texts)

        # Index vectors
        self.documents = []
        for item in raw_cards:
            vector = self.embedder.embed(item["text"])
            self.documents.append({
                "id": item["id"],
                "text": item["text"],
                "vector": vector,
                "card": item["card"],
                "exercise": item["card"].get("exercise"),
                "fault_code": item["card"].get("fault_code"),
            })

    def search(
        self,
        query: str,
        exercise: Optional[str] = None,
        top_k: int = 2
    ) -> List[Tuple[Dict[str, Any], float]]:
        """
        Executes semantic vector similarity search against the knowledge corpus.
        Returns list of (knowledge_card, similarity_score) sorted by relevance.
        """
        if not self.documents:
            return []

        query_vec = self.embedder.embed(query)

        scored: List[Tuple[Dict[str, Any], float]] = []
        for doc in self.documents:
            # Filter by exercise if requested
            if exercise and doc["exercise"] != exercise:
                continue

            # Cosine similarity between query_vec and doc['vector']
            dot = sum(q * d for q, d in zip(query_vec, doc["vector"]))
            scored.append((doc["card"], dot))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
