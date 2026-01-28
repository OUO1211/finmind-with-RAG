"""
RAGService：RAG 服務
功能：整合檢索（Retrieval）和生成（Generation），回答使用者問題
"""

import requests
from .vector_store import VectorStore


class RAGService:
    """
    RAG 服務

    職責：
    1. 收到使用者問題
    2. 用 VectorStore 找相關 chunks
    3. 組成 prompt（問題 + chunks）
    4. 呼叫 Ollama LLM 生成回答
    """

    def __init__(self, vector_store: VectorStore,
                 model: str = "gemma3:12b",
                 base_url: str = "http://localhost:11434"):
        """
        建構子

        Args:
            vector_store: VectorStore 實例
            model: LLM 模型名稱
            base_url: Ollama API 網址
        """
        self.vector_store = vector_store
        self.model = model
        self.base_url = base_url

    def ask(self, question: str, stock_id: str = None) -> str:
        """
        回答使用者問題

        Args:
            question: 使用者的問題

        Returns:
            LLM 生成的回答
        """
        # 1. 搜尋相關 chunks
        chunks = self.vector_store.query(question, stock_id = stock_id)

        # 2. 組成 prompt
        context = "\n".join(chunks[0])
        prompt = f"""根據以下資料回答問題：

        {context}

        問題：{question}
        """

        # 3. 呼叫 Ollama API
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )
        result = response.json()
        answer = result["response"]

        # 4. 回傳答案
        return answer
