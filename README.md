
# 利用大語言模型與 GraphRAG 技術提升心電圖診斷與報告生成的準確性與穩定性
> **國科會研究計畫 - 關鍵技術三：基於 GraphRAG 的知識圖譜與檢索增強摘要生成**

本專案旨在結合大型語言模型（LLMs）與圖檢索增強生成（GraphRAG）技術，針對結構化與非結構化之心電圖（ECG）臨床文本進行知識圖譜建構。透過實體（Entities）與關係（Relations）的抽取，建立領域專屬的知識網絡，藉此大幅提升 AI 在心電圖自動化診斷與醫學報告生成上的準確性、邏輯穩定性，並有效抑制大模型的「幻覺（Hallucination）」問題。

---

## 📌 核心架構與技術特點

傳統的 RAG（檢索增強生成）僅依賴純文字片段的向量相似度比對，容易在面對跨文本、複雜醫學邏輯推導時斷章取義。本專案採用的 **GraphRAG** 架構具備以下優勢：

1. **圖譜驅動檢索**：將心電圖特徵（如：ST節段上升、QRS波群寬度、T波倒置）與醫學診斷（如：急性心肌梗塞、心房顫動）轉化為圖節點與邊，保留完整的臨床因果脈絡。
2. **多層次摘要生成**：利用 Community Detection（社群偵測）演算法對圖譜進行全局摘要，確保在生成最終心電圖臨床報告時，能同時兼顧「局部診斷特徵」與「全局病歷脈絡」。
3. **評估自動化**：專案整合了 Ragas 評估框架，針對生成報告的忠實度（Faithfulness）、答案相關性（Answer Relevance）以及上下文召回率（Context Recall）進行量化基準測試（Baseline Evaluation）。

---

## 📂 專案目錄結構

```text
graghRAG_ECG/
├── api_service.py              # 基於 FastAPI 架設的 GraphRAG 檢索與報告生成 API 服務
├── prepare_samples.py          # 測試資料集預處理與格式轉換腳本
├── run_evaluation.py           # 整合 Ragas 評估框架的自動化測試主程式
├── run_evaluation_baseline.py  # 傳統 RAG / 基礎 LLM 的對照組評估腳本
├── samples_for_ragas.json      # 用於 Ragas 評估的黃金標準（Gold Standard）測試資料集
├── settings.yaml               # GraphRAG 核心引擎設定檔（LLM/Embedding 參數、Chunk 大小）
├── .gitignore                  # Git 忽略清單（已隔離密鑰、本地快取與 Miniconda 環境）
└── README.md                   # 本說明文件

```

---

## 🛠️ 環境架構與安裝指南

本專案執行於實驗室遠端 Linux 伺服器環境，並透過 **Miniconda** 進行 Python 虛擬環境隔離。

### 1. 複製專案並進入目錄

```bash
git clone [https://github.com/Kai-1907/graghRAG_ECG.git](https://github.com/Kai-1907/graghRAG_ECG.git)
cd graghRAG_ECG

```

### 2. 建立並啟用 Conda 環境

```bash
# 建立專案專屬環境（以 Python 3.10 為例）
conda create -n ecg_rag python=3.10 -y
conda activate ecg_rag

# 安裝 GraphRAG、FastAPI 與 Ragas 相關依賴套件
pip install graphrag fastapi uvicorn ragas langchain openai

```

### 3. 配置環境變數 `.env`

為了安全起見，API 金鑰已被排除於版本控制之外。請在專案根目錄下建立 `.env` 檔案，並填入您的 API Key：

```ini
GRAGHRAG_API_KEY=your_openai_or_gemini_api_key_here
# 若使用本地 Ollama 或自建模型終端，請依需求配置：
# GRAGHRAG_API_BASE=http://localhost:11434/v1

```

---

## 🚀 執行與使用說明

### 步驟一：構建 GraphRAG 知識圖譜（Indexing）

在確保 `settings.yaml` 設定正確，且 `input/` 資料夾中已放置心電圖文本資料後，執行以下指令建立索引：

```bash
python -m graphrag.index --root .

```

> 💡 *註：索引生成後會產生巨大的 `artifacts/` 與 `output/` 資料夾，該部分已被 `.gitignore` 安全忽略，不會重複上傳。*

### 步驟二：啟動後端 API 服務

執行 `api_service.py` 啟動 FastAPI 伺服器，對外提供即時的心電圖報告檢索與生成接口：

```bash
python api_service.py

```

預設會於 `http://localhost:8000` 啟動。可透過發送 POST 請求至 `/query` 端點進行全域（Global）或區域（Local）的 GraphRAG 檢索。

### 步驟三：執行 Ragas 效能評估測試

為了客觀衡量 GraphRAG 提升的穩定度，專案提供自動化評估腳本：

```bash
# 執行 GraphRAG 實驗組評估
python run_evaluation.py

# 執行傳統 Baseline 對照組評估
python run_evaluation_baseline.py

```

評估結果將自動計算 **Faithfulness（忠實度）** 與 **Answer Relevance（答案相關度）**，作為國科會計畫結案報告之數據支持。

---

## 👥 團隊協作與計畫交接

* **計畫負責人**：仕凱（負責 關鍵技術三：GraphRAG 架構設計、知識圖譜構建與 API 開發）
* **團隊協作者**：敏心（負責 關鍵技術二之初步結果對接與後續系統整合）

### 📢 交接注意事項：

1. **資料夾權限**：伺服器上的 `cache/` 與 `output/` 目錄含有本地快取的圖索引，交接時建議直接於伺服器內部分享目錄權限，切勿透過 Git 傳輸。
2. **測試數據**：`samples_for_ragas.json` 內含醫療去識別化之模擬文本，修改評估範例時請確保符合學術倫理規範。

```

