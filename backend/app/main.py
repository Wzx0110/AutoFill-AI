from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.core.config import settings
from app.services.llm_service import llm_service

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# 定義 Request Schema
class ChatRequest(BaseModel):
    message: str

@app.get("/")
def health_check():
    return {"status": "ok", "service": "AutoFill AI"}

# 測試接口
@app.post("/api/test-llm")
def test_llm(request: ChatRequest):
    """
    測試 LLM 是否連線成功
    """
    try:
        # 呼叫服務
        reply = llm_service.generate_response(request.message)
        return {"reply": reply}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)