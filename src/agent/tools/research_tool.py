from typing import List, Dict, Any, Optional
import os
import json
import logging
import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
try:
    from duckduckgo_search import DDGS
except ImportError:
    from ddgs import DDGS

from src.data.database import DatabaseManager
from configs.settings import settings

logger = logging.getLogger(__name__)


class ResearchTool:
    def __init__(self, knowledge_base_dir: Optional[str] = None):
        self.kb_dir = Path(knowledge_base_dir) if knowledge_base_dir else Path(settings.knowledge_base_dir)
        self.kb_dir.mkdir(parents=True, exist_ok=True)
        
        # Use local embeddings (Ollama or sentence-transformers)
        self.embeddings = self._init_embeddings()
        
        persist_dir = self.kb_dir / 'chroma'
        self.vector_store = None
        if persist_dir.exists():
            try:
                self.vector_store = Chroma(
                    persist_directory=str(persist_dir), 
                    embedding_function=self.embeddings
                )
                logger.info("Loaded existing knowledge base from Chroma")
            except Exception as e:
                logger.warning(f"Failed to load existing vector store: {e}")

        try:
            self.ddgs = DDGS()
        except Exception as e:
            logger.warning(f"Failed to initialize DuckDuckGo search: {e}")
            self.ddgs = None

        # Auto-initialize knowledge base with seed agricultural documents if empty
        self._init_seed_knowledge()
    
    def _init_embeddings(self):
        """Initialize embeddings using Ollama or fallback to sentence-transformers."""
        try:
            import httpx
            with httpx.Client(timeout=3.0) as client:
                resp = client.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
                if resp.status_code == 200:
                    emb_resp = client.post(
                        f"{settings.ollama_base_url}/api/embed",
                        json={"model": settings.ollama_model, "input": "test"},
                        timeout=3.0,
                    )
                    if emb_resp.status_code == 200:
                        from langchain_ollama import OllamaEmbeddings
                        embeddings = OllamaEmbeddings(
                            model=settings.ollama_model,
                            base_url=settings.ollama_base_url,
                        )
                        logger.info("Using Ollama embeddings")
                        return embeddings
                    logger.warning(f"Ollama embeddings not supported (status {emb_resp.status_code}), falling back to sentence-transformers")
        except Exception as e:
            logger.warning(f"Ollama check failed: {e}, falling back to sentence-transformers")
        
        try:
            from sentence_transformers import SentenceTransformer
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            model = SentenceTransformer('all-MiniLM-L6-v2', local_files_only=True)
            
            # Create a wrapper class to match LangChain interface
            class SentenceTransformerEmbeddings:
                def __init__(self, model):
                    self.model = model
                
                def embed_documents(self, texts: List[str]) -> List[List[float]]:
                    return self.model.encode(texts).tolist()
                
                def embed_query(self, text: str) -> List[float]:
                    return self.model.encode([text])[0].tolist()
            
            logger.info("Using sentence-transformers embeddings")
            return SentenceTransformerEmbeddings(model)
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            raise

    def _init_seed_knowledge(self):
        """Auto-populate knowledge base with seed agricultural documents if empty."""
        if self.vector_store is None:
            try:
                self.vector_store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=str(self.kb_dir / 'chroma')
                )
            except Exception as e:
                logger.warning(f"Failed to create vector store: {e}")
                return

        # Check if the vector store already has documents
        try:
            existing = self.vector_store._collection.count()
            if existing > 0:
                logger.info(f"Knowledge base has {existing} documents")
                return
        except Exception:
            pass

        seed_docs = [
            ("Plant Disease Identification and Management",
             "Common tomato diseases include: Early blight (Alternaria solani) causes brown spots with concentric rings on lower leaves. Late blight (Phytophthora infestans) causes water-soaked lesions that turn brown. Bacterial spot (Xanthomonas) causes small water-soaked spots. Septoria leaf spot causes small circular spots with dark margins. Treatment: remove infected leaves, apply copper-based fungicides, ensure proper spacing for air circulation."),
            ("Tomato Yellow Leaf Curl Virus",
             "TYLCV symptoms include upward curling of leaves, yellowing of leaf veins, stunted growth, and reduced fruit production. The virus is transmitted by whiteflies. Management: control whitefly populations with insecticides or yellow sticky traps, use resistant tomato varieties (e.g., 'Cleopatra', 'Tyche'), remove infected plants immediately."),
            ("Spider Mite Management",
             "Spider mites cause stippling on leaves, webbing, and leaf drop. They thrive in hot, dry conditions. Management: introduce natural predators (lady beetles, lacewings), use neem oil or insecticidal soap, maintain adequate humidity, prune heavily infested leaves, rotate miticides to prevent resistance."),
            ("Tomato Mosaic Virus",
             "Tobacco mosaic virus (TMV) in tomatoes causes mottled leaves, stunted growth, and distorted fruit. Transmission is mechanical via contaminated tools or hands. Management: wash hands and tools between plants, use virus-free seedlings, apply copper-based fungicides preventatively, remove infected leaves promptly."),
            ("Integrated Pest Management (IPM) Strategies",
             "IPM combines biological, cultural, physical, and chemical tools. Key strategies: monitor pest populations regularly, use pheromone traps, plant resistant varieties, practice crop rotation, maintain beneficial insects, apply pesticides only when economic thresholds are reached, and use targeted rather than broad-spectrum treatments."),
            ("Soil Health Management",
             "Healthy soil should have proper pH (6.0-6.8 for tomatoes), adequate organic matter (>3%), good drainage, and balanced nutrient levels. Test soil annually. Add compost or well-rotted manure. Practice cover cropping. Avoid over-tilling which damages soil structure. Rotate crops to prevent nutrient depletion and disease buildup."),
            ("Weather Impact on Crop Health",
             "High temperatures (above 35C) cause heat stress, reducing pollination and fruit set. Low humidity increases transpiration. Excessive rain promotes fungal diseases. Wind above 20 km/h can cause physical damage. Irrigate early morning to reduce disease pressure. Use mulch to retain soil moisture and regulate temperature."),
            ("Irrigation Best Practices",
             "Tomatoes need 25-30mm of water per week. Water deeply and infrequently to encourage deep root growth. Drip irrigation is preferred over overhead to reduce leaf wetness and disease risk. Water in the morning. Avoid wetting foliage. Mulch to reduce evaporation. Adjust irrigation based on weather forecasts."),
            ("Fertilization Guidelines for Tomatoes",
             "Tomatoes need balanced fertilization: nitrogen (N) for vegetative growth, phosphorus (P) for root and fruit development, potassium (K) for overall health. Use a 4-4-8 or 5-10-10 fertilizer. Start with half-strength at planting, increase as plants grow. Side-dress with compost after first fruit set. Avoid excess nitrogen which promotes leaf growth over fruit."),
            ("Nutrient Deficiency Symptoms",
             "Nitrogen deficiency: yellowing of older leaves. Phosphorus deficiency: purple or dark green leaves, stunted growth. Potassium deficiency: yellowing between leaf veins, leaf edges curl. Calcium deficiency: blossom end rot on fruits. Magnesium deficiency: interveinal yellowing on older leaves. Iron deficiency: interveinal chlorosis on young leaves."),
        ]

        try:
            docs = [Document(page_content=content, metadata={"source": "seed_knowledge", "topic": title}) for title, content in seed_docs]
            self.vector_store.add_documents(docs)
            logger.info(f"Auto-initialized knowledge base with {len(seed_docs)} seed documents")
        except Exception as e:
            logger.warning(f"Failed to seed knowledge base: {e}")

    def search_knowledge_base(self, query: str, k: int = 5) -> str:
        """Search the local knowledge base using RAG. Falls back to web search if KB is empty."""
        if not self.vector_store:
            return self.search_web(f"agriculture {query}", max_results=k)
        
        try:
            docs = self.vector_store.similarity_search(query, k=k)
            if not docs:
                return self.search_web(f"agriculture {query}", max_results=k)
            
            results = []
            for i, doc in enumerate(docs, 1):
                results.append(f"[Source {i}] {doc.page_content}")
                if hasattr(doc, 'metadata') and doc.metadata:
                    results.append(f"  Metadata: {doc.metadata}")
            
            return "\n\n".join(results)
        except Exception as e:
            logger.error(f"Knowledge base search failed: {e}")
            return self.search_web(f"agriculture {query}", max_results=k)

    def search_web(self, query: str, max_results: int = 5) -> str:
        """Search the web using DuckDuckGo."""
        if not self.ddgs:
            return f"Web search unavailable - DuckDuckGo client not initialized"
        try:
            results = self.ddgs.text(query, max_results=max_results)
            if not results:
                return f"No web search results found for: {query}"

            snippets = []
            for i, r in enumerate(results, 1):
                snippets.append(f"[{i}] {r['title']}")
                snippets.append(f"    {r['body']}")
                snippets.append(f"    Source: {r['link']}")
                snippets.append("")

            return "\n".join(snippets)
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return f"Web search failed: {str(e)}"
    
    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Add a document to the knowledge base."""
        try:
            if not self.vector_store:
                # Initialize new vector store
                self.vector_store = Chroma(
                    embedding_function=self.embeddings,
                    persist_directory=str(self.kb_dir / 'chroma')
                )
            
            doc = Document(page_content=content, metadata=metadata or {})
            self.vector_store.add_documents([doc])
            # Chroma 0.4+ auto-persists; no explicit .persist() needed
            
            logger.info(f"Document added to knowledge base")
            return "Document successfully added to knowledge base"
        except Exception as e:
            logger.error(f"Failed to add document: {e}")
            return f"Failed to add document: {str(e)}"
    
    def add_documents_from_directory(self, directory: str) -> str:
        """Add all text/markdown files from a directory to the knowledge base."""
        try:
            dir_path = Path(directory)
            if not dir_path.exists():
                return f"Directory not found: {directory}"
            
            text_files = list(dir_path.glob("*.txt")) + list(dir_path.glob("*.md"))
            if not text_files:
                return f"No text or markdown files found in {directory}"
            
            added_count = 0
            for file_path in text_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    metadata = {
                        'source': str(file_path),
                        'filename': file_path.name,
                        'file_hash': hashlib.md5(content.encode()).hexdigest()
                    }
                    
                    result = self.add_document(content, metadata)
                    if "successfully" in result:
                        added_count += 1
                except Exception as e:
                    logger.error(f"Failed to process {file_path}: {e}")
            
            return f"Added {added_count} documents from {directory} to knowledge base"
        except Exception as e:
            logger.error(f"Failed to add documents from directory: {e}")
            return f"Failed to add documents from directory: {str(e)}"
    
    def search_agricultural_info(self, topic: str) -> str:
        """Specialized search for agricultural information combining KB and web."""
        kb_results = self.search_knowledge_base(topic, k=3)
        web_results = self.search_web(f"agriculture {topic}", max_results=3)
        
        combined = f"KNOWLEDGE BASE RESULTS:\n{kb_results}\n\n" \
                   f"WEB SEARCH RESULTS:\n{web_results}"
        return combined