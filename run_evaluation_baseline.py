import os
import pandas as pd
import subprocess
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, answer_correctness
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# ---------------------------------------------------------
# 1. 基礎設定：完全本地化類別
# ---------------------------------------------------------
# 定義本地 LLM (負責評分)
local_llm = ChatOllama(
    model="llama3",
    base_url="http://127.0.0.1:11434"
)

# 定義本地 Embedding (負責計算向量相關指標)
# 使用專門類別可以避免 400 Invalid Input Type 錯誤
local_embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://127.0.0.1:11434"
)

# 測試樣本
test_samples = [
    {"input": "NORM: 80.0, SBRAD: 0.0", "reference": "正常心電圖，無竇性心率過緩。"},
    {"input": "NORM: 0.0, AFIB: 100.0", "reference": "心電圖顯示心房顫動。"},
    {"input": "NORM: 0.0, STTC: 100.0", "reference": "心電圖顯示 ST-T 段改變，可能存在心肌缺血。"},
    {"input": "NORM: 0.0, MI: 100.0", "reference": "心電圖顯示心肌梗塞跡象。"},
    {"input": "NORM: 50.0, HYP: 50.0", "reference": "正常心律伴隨左心室肥大可能。"},
    {"input": "NORM: 0.0, SB: 100.0", "reference": "心電圖顯示竇性心率過緩。"},
    {"input": "NORM: 0.0, SR: 100.0", "reference": "竇性心律正常。"},
    {"input": "NORM: 0.0, STACH: 100.0", "reference": "心電圖顯示竇性心率過快。"},
    {"input": "NORM: 0.0, PVC: 100.0", "reference": "心電圖顯示心室早期收縮。"},
    {"input": "NORM: 100.0, ALL: 0.0", "reference": "完全正常的心電圖結果。"},
]

# ---------------------------------------------------------
# 2. 定義呼叫 GraphRAG 的函式 (保持不變)
# ---------------------------------------------------------
def call_graphrag(question):
    graphrag_path = "/home/g114056107/miniconda3/envs/graph_env/bin/graphrag"
    cmd = [
        graphrag_path, "query",
        "--root", ".",
        "--method", "global",
        "--query", question
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"\n❌ GraphRAG 執行失敗: {e.stderr}")
        return None

# ---------------------------------------------------------
# 3. 執行生成階段
# ---------------------------------------------------------
print("🚀 開始執行純 LLM (Baseline) 生成測試樣本...")
evaluation_data = []

for sample in tqdm(test_samples):
    # 改成直接對 LLM 提問，不經過 GraphRAG 檢索
    # 使用 invoke 獲取回應物件，再提取其中的 content
    raw_response = local_llm.invoke(sample["input"])
    response = raw_response.content 
    
    evaluation_data.append({
        "question": sample["input"],
        "contexts": [""],           # 關鍵：給空字串，證明它沒看參考資料，分數才會掉下來
        "answer": response,
        "ground_truth": sample["reference"]
    })

# ---------------------------------------------------------
# 4. 執行本地 RAGAS 評估階段
# ---------------------------------------------------------
if evaluation_data:
    print(f"\n✅ 已完成 {len(evaluation_data)} 筆樣本生成。")
    print("🧪 正在使用 Ollama 專用介面進行評估...")
    
    eval_dataset = Dataset.from_list(evaluation_data)
    
    try:
        result = evaluate(
            dataset=eval_dataset,
            metrics=[faithfulness, answer_relevancy, answer_correctness],
            llm=local_llm,
            embeddings=local_embeddings
        )
        
        print("\n" + "="*30)
        print("📊 最終本地評估結果")
        print("="*30)
        print(result)
        
        # 儲存結果
        df_results = result.to_pandas()
        output_file = 'evaluation_results.csv'
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n💾 成功！數據已儲存至: {output_file}")
        
    except Exception as e:
        print(f"\n❌ 評估過程發生錯誤: {str(e)}")
else:
    print("\n❌ 未能生成任何數據。")