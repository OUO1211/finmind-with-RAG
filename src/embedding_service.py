"""
EmbeddingService：向量化服務
功能：將文字轉換成向量（embedding），使用 Ollama API
"""

import requests


class EmbeddingService:
    """
    向量化服務

    職責：
    1. 連接 Ollama API
    2. 將文字轉換成向量
    """

    def __init__(self, model: str = "nomic-embed-text",
                 base_url: str = "http://localhost:11434"):
        """
        建構子

        Args:
            model: 模型名稱，預設 nomic-embed-text
            base_url: Ollama API 網址
        """
        self.model = model
        self.base_url = base_url

    def embed(self, text: str) -> list[float]:
        """
        將文字轉換成向量

        Args:
            text: 要轉換的文字

        Returns:
            向量（list of floats）
        """
        url = f"{self.base_url}/api/embeddings"
        response = requests.post(
            url,
            json={
                "model": self.model,
                "prompt": text
            }
        )
        result = response.json()
        return result["embedding"]
