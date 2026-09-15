"""
RAGRetriever — build and query a RAG knowledge base from text documents.

Embeddings priority:
  1. langchain_ollama.OllamaEmbeddings (best quality, requires Ollama running)
  2. sentence_transformers via a LangChain-compatible wrapper (offline fallback)
"""

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_core.embeddings import Embeddings
from pathlib import Path
from configs.settings import settings
import logging
from typing import List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sentence-Transformer LangChain wrapper
# ---------------------------------------------------------------------------
class _STEmbeddings(Embeddings):
    """LangChain-compatible wrapper around sentence_transformers."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer  # lazy import
        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._model.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> List[float]:
        return self._model.encode([text], show_progress_bar=False)[0].tolist()


# ---------------------------------------------------------------------------
# RAGRetriever
# ---------------------------------------------------------------------------
class RAGRetriever:
    """Build and query a RAG knowledge base from documents (FAO, USDA, etc.)."""

    def __init__(self, persist_dir: str = None):
        self.persist_dir = Path(persist_dir or settings.knowledge_base_dir) / "chroma"
        self.embeddings = self._init_embeddings()
        self.vector_store = None
        if self.persist_dir.exists():
            try:
                self.vector_store = Chroma(
                    persist_directory=str(self.persist_dir),
                    embedding_function=self.embeddings,
                )
                logger.info("Loaded existing Chroma KB from %s", self.persist_dir)
            except Exception as e:
                logger.warning("Could not load Chroma KB: %s", e)

    # ------------------------------------------------------------------
    def _init_embeddings(self) -> Embeddings:
        try:
            from langchain_ollama import OllamaEmbeddings
            emb = OllamaEmbeddings(
                model=settings.ollama_model,
                base_url=settings.ollama_base_url,
            )
            # quick connectivity check
            emb.embed_query("ping")
            logger.info("RAGRetriever: using Ollama embeddings")
            return emb
        except Exception as e:
            logger.warning("Ollama embeddings unavailable (%s); falling back to sentence-transformers", e)

        return _STEmbeddings()

    # ------------------------------------------------------------------
    def index_documents(self, docs_dir: str):
        """Load .txt and .md files from directory and (re-)build the index."""
        dir_path = Path(docs_dir)
        files = list(dir_path.glob("**/*.txt")) + list(dir_path.glob("**/*.md"))
        if not files:
            logger.warning("No .txt/.md files found in %s", docs_dir)
            return

        documents = []
        for fp in files:
            try:
                loader = TextLoader(str(fp), encoding="utf-8")
                documents.extend(loader.load())
            except Exception as e:
                logger.warning("Could not load %s: %s", fp, e)

        if not documents:
            return

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(documents)

        self.vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=str(self.persist_dir),
        )
        # Chroma 0.4+ auto-persists; no explicit .persist() call needed
        logger.info("Indexed %d chunks from %d docs into %s", len(chunks), len(documents), self.persist_dir)

    # ------------------------------------------------------------------
    def retrieve(self, query: str, k: int = 5) -> str:
        if not self.vector_store:
            return "Knowledge base not indexed yet."
        docs = self.vector_store.similarity_search(query, k=k)
        if not docs:
            return f"No relevant documents found for: {query}"
        return "\n\n".join(d.page_content for d in docs)