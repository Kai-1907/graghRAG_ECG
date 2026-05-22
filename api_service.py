from fastapi import FastAPI, Query
import subprocess
import uvicorn

app = FastAPI(title="Medical GraphRAG API Service")

@app.get("/query")
def run_graphrag_query(user_query: str = Query(..., description="輸入想查詢的醫學問題"), method: str = "global"):
    """
    呼叫本地的 GraphRAG 進行檢索
    """
    try:
        # 建立執行指令，這就是妳在終端機敲的那行
        cmd = [
            "graphrag", "query",
            "--root", ".",
            "--method", method,
            "--query", user_query
        ]
        
        # 執行指令並獲取輸出
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        if result.returncode == 0:
            return {"status": "success", "data": result.stdout}
        else:
            return {"status": "error", "message": result.stderr}
            
    except Exception as e:
        return {"status": "exception", "message": str(e)}

if __name__ == "__main__":
    # 啟動服務，埠號設定為 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)