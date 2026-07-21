"""
FinMind RAG 財報分析系統

模組說明：
- CacheManager: 快取管理
- DataFetcher: FinMind API 呼叫
- DataService: 資料服務層（整合快取 + API）
- TextProcessor: 文字處理（DataFrame → chunks）
- EmbeddingService: 向量化服務（Ollama）
- VectorStore: 向量資料庫（ChromaDB）
- RAGService: RAG 問答服務
"""

from importlib import import_module

__all__ = [
    "CacheManager",
    "DataFetcher",
    "DataService",
    "TextProcessor",
    "EmbeddingService",
    "VectorStore",
    "RAGService",
]

_MODULE_BY_NAME = {
    "CacheManager": "cache_manager",
    "DataFetcher": "data_fetcher",
    "DataService": "data_service",
    "TextProcessor": "text_processor",
    "EmbeddingService": "embedding_service",
    "VectorStore": "vector_store",
    "RAGService": "rag_service",
}


def __getattr__(name):
    module_name = _MODULE_BY_NAME.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    return getattr(module, name)
