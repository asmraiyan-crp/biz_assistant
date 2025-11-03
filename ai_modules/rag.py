import numpy as np

try:
    from sentence_transformers import SentenceTransformer

    USE_ST = True
except Exception:
    USE_ST = False
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


class SimpleRAG:
    def __init__(self, documents=None):
        self.documents = documents or []
        self.embeddings = None
        self.tfidf = None

        if USE_ST:
            try:
                self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            except Exception:
                print("Warning: Failed to load SentenceTransformer, falling back to TF-IDF.")
                self.embedder = None
                self.vectorizer = TfidfVectorizer(stop_words='english')
        else:
            self.embedder = None
            self.vectorizer = TfidfVectorizer(stop_words='english')

    def index(self):
        if not self.documents:
            print("Warning: No documents to index.")
            return

        texts = [d['text'] for d in self.documents]

        if USE_ST and self.embedder:
            self.embeddings = self.embedder.encode(texts, convert_to_numpy=True)
        else:
            if texts:
                self.tfidf = self.vectorizer.fit_transform(texts)

    def add(self, doc):
        """Adds a document, but does not re-index. Call index() again."""
        self.documents.append(doc)
        # Note: In a real system, you'd want an incremental index
        # For this simple class, user must call .index() again.
        self.embeddings = None  # Invalidate old index
        self.tfidf = None  # Invalidate old index

    def query(self, q, top_k=3):
        # ROBUSTNESS FIX: Ensure index has been called
        if self.embeddings is None and self.tfidf is None:
            if not self.documents:
                return []  # No documents, return empty list
            print("Warning: .index() has not been called. Indexing now.")
            self.index()
            # If still None (e.g., no documents), return
            if self.embeddings is None and self.tfidf is None:
                return []

        if USE_ST and self.embedder and self.embeddings is not None:
            qv = self.embedder.encode([q], convert_to_numpy=True)
            sims = (self.embeddings @ qv.T).squeeze()
            # Handle case where self.embeddings is 1D (only one doc indexed)
            if sims.ndim == 0:
                sims = np.array([sims])

            idx = sims.argsort()[::-1][:top_k]
            return [self.documents[i] for i in idx if sims[i] > 0]
        else:
            # Fallback to TF-IDF
            qv = self.vectorizer.transform([q])
            sims = linear_kernel(qv, self.tfidf).flatten()
            idx = sims.argsort()[::-1][:top_k]
            return [self.documents[i] for i in idx if sims[i] > 0]