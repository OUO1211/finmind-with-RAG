"""
FinMind RAG 財報分析系統 - 主程式
"""

import os
import time
import pandas as pd
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
    embedding_service = EmbeddingService(model="BAAI/bge-base-zh-v1.5")


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

    stock_list = data_service.fetcher.get_stock_list()

    target_ids = ["2330", "2317"]
    target_stocks = list({s["stock_id"]: s for s in stock_list if s["stock_id"] in target_ids}.values())


    for stock in target_stocks:
        stock_id = stock["stock_id"]
        stock_name = stock["stock_name"]

        df = data_service.get_data(
            stock_id=stock_id,
            data_type="financial_statement",
            start_date="2022-01-01",
            end_date="2025-01-01"
        )

        all_chunks = []

        if not df.empty:
            # 轉換成文字 chunks
            chunks = text_processor.df_to_chunks(df, stock_name=stock_name)
            all_chunks.extend(chunks)
        
        time.sleep(1)


        per_df = data_service.get_data(
            stock_id=stock_id,
            data_type="per",
            start_date="2023-01-01",
            end_date="2024-01-01"
        )
        if not per_df.empty:
            per_df['date'] = pd.to_datetime(per_df['date'])  # 轉成日期格式
            per_df = per_df.groupby(per_df['date'].dt.to_period('M')).last()  # 每月最後一筆
            per_df['date'] = per_df['date'].dt.strftime('%Y-%m-%d')
            per_df = per_df.reset_index(drop=True)  # 重設 index

            per_chunks = text_processor.per_to_chunks(per_df, stock_name)
            all_chunks.extend(per_chunks)

        balance_sheet_df = data_service.get_data(
            stock_id=stock_id,
            data_type="balance_sheet",
            start_date="2022-01-01",
            end_date="2025-01-01"
        )
        if not balance_sheet_df.empty:
            balance_sheet_chunks = text_processor.df_to_chunks(balance_sheet_df, stock_name)

        
            all_chunks.extend(balance_sheet_chunks)

            # 測試 ROE/ROA
            roe_roa_chunks = text_processor.roe_roa_to_chunks(df, balance_sheet_df, stock_name)
            print(f"ROE/ROA chunks: {roe_roa_chunks[:3]}") 
            all_chunks.extend(roe_roa_chunks)

            # 測試負債比/流動比率
            debt_chunks = text_processor.debt_current_to_chunks(balance_sheet_df, stock_name)
            print(f"Debt/Current chunks: {debt_chunks[:3]}")
            all_chunks.extend(debt_chunks)

            # 毛利率/營益率
            margin_chunks = text_processor.margin_to_chunks(df, stock_name)
            print(f"Margin chunks: {margin_chunks[:3]}")
            all_chunks.extend(margin_chunks)



        if all_chunks:
            vector_store.add(all_chunks, stock_id)

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

        filter_choice = input("是否要篩選特定股票？(輸入股票代號，或按 Enter 跳過)：").strip()
        n_input = input("要幾筆資料？(預設3，按 Enter 跳過)：").strip()
        n_results = int(n_input) if n_input else 3
        if filter_choice:
            stock_id = filter_choice
        else:
            stock_id = None
        answer = rag.ask(question, stock_id, n_results)
        print(f"\n回答：\n{answer}")


if __name__ == "__main__":
    main()
