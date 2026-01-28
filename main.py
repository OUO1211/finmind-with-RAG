"""
FinMind RAG 財報分析系統 - 主程式
"""

import os
from dotenv import load_dotenv
from src import (
    DataService,
    TextProcessor,
    EmbeddingService,
    VectorStore,
    RAGService,
)


def main():
    # 載入環境變數
    load_dotenv()

    print("=" * 60)
    print("FinMind RAG 財報分析系統")
    print("=" * 60)

    # --- 1. 初始化服務 ---
    print("\n[1/5] 初始化資料服務...")
    data_service = DataService(cache_dir="data")

    print("\n[2/5] 初始化文字處理器...")
    text_processor = TextProcessor()

    print("\n[3/5] 初始化向量化服務...")
    embedding_service = EmbeddingService(model="nomic-embed-text")

    print("\n[4/5] 初始化向量資料庫...")
    vector_store = VectorStore(
        persist_path="./chroma_db",
        collection_name="financial_reports",
        embedding_service=embedding_service
    )

    print("\n[5/5] 初始化 RAG 服務...")
    rag = RAGService(
        vector_store=vector_store,
        model="gemma3:12b"
    )

    print("\n系統初始化完成！")

    # --- 2. 載入資料（示範：台積電） ---
    print("\n" + "=" * 60)
    print("載入股票資料")
    print("=" * 60)

    stock_id = "2330"
    stock_name = "台積電"

    df = data_service.get_data(
        stock_id=stock_id,
        data_type="financial_statement",
        start_date="2023-01-01",
        end_date="2024-01-01"
    )

    if not df.empty:
        # 轉換成文字 chunks
        chunks = text_processor.df_to_chunks(df, stock_name=stock_name)
        print(f"產生 {len(chunks)} 個文字片段")

        # 存入向量資料庫（如果還沒存過）
        # 注意：重複執行會重複存入，實際應用需要檢查
        # vector_store.add(chunks=chunks, stock_id=stock_id)

    # --- 3. 互動問答 ---
    print("\n" + "=" * 60)
    print("RAG 問答系統（輸入 'exit' 離開）")
    print("=" * 60)

    while True:
        question = input("\n請輸入問題：").strip()

        if question.lower() == "exit":
            print("再見！")
            break

        if not question:
            continue

        print("\n思考中...")
        answer = rag.ask(question)
        print(f"\n回答：\n{answer}")


if __name__ == "__main__":
    main()
