"""
EmbeddingService：向量化服務
功能：將文字轉換成向量（embedding），使用 HuggingFace sentence-transformers
"""


from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """
    向量化服務

    職責：
    1. 載入 HuggingFace embedding model
    2. 將文字轉換成向量
    """

    def __init__(self, model: str = "BAAI/bge-base-zh-v1.5"):
        """
        建構子

        Args:
            model: HuggingFace 模型名稱，預設使用中文模型
        """
        self.model = SentenceTransformer(model)


    def embed(self, text: str) -> list[float]:
        """將文字轉換成向量"""
        embedding = self.model.encode(text)
        return embedding.tolist()  # 轉成 list

    

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批次轉換"""
        embeddings = self.model.encode(texts)
        return embeddings.tolist()

        

