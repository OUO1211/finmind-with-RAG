"""
VectorStore：向量資料庫
功能：使用 ChromaDB 儲存和查詢向量
"""

import chromadb
from .embedding_service import EmbeddingService


class VectorStore:
    """
    向量資料庫

    職責：
    1. 連接 ChromaDB（持久化模式）
    2. 將文字 chunks + 向量存入資料庫
    3. 根據查詢向量，找出最相似的 chunks
    """

    def __init__(self, persist_path: str, collection_name: str = "defaultCollection",
                 embedding_service: EmbeddingService = None):
        """
        建構子

        Args:
            persist_path: 資料庫儲存路徑
            collection_name: Collection 名稱
            embedding_service: EmbeddingService 實例
        """
        self.client = chromadb.PersistentClient(path=persist_path)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self.embedding_service = embedding_service

    def add(self, chunks: list[str], stock_id: str):
        """
        將多個文字 chunks 存入 ChromaDB

        Args:
            chunks: 文字列表
            stock_id: 股票代號
        """
        existing = self.collection.get(where={"stock_id": stock_id})
        if existing["ids"]:
            self.collection.delete(ids=existing["ids"])
            print(f"[VectorStore] 已刪除 {stock_id} 的 {len(existing['ids'])} 筆舊資料")


        ids = []
        embeddings = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            id = f"{stock_id}_{i}"
            ids.append(id)

            embed = self.embedding_service.embed(chunk)
            embeddings.append(embed)

            metadata = {"stock_id": stock_id}
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )
        print(f"[VectorStore] 已存入 {len(chunks)} 個 chunks，stock_id={stock_id}")

    def query(self, text: str, n_results: int = 3, stock_id: str = None):
        """
        搜尋最相似的 chunks

        Args:
            text: 查詢文字
            n_results: 回傳幾筆結果
            stock_id: 可選，只搜尋特定股票

        Returns:
            找到的文字列表（二維 list）
        """
        embedding = self.embedding_service.embed(text)

        if stock_id:
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
                where={"stock_id": stock_id}
            )
        else:
            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=n_results,
            )

        return result["documents"]
