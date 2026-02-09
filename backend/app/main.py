from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel
from typing import List

from app.core.config import settings
from app.services.llm_service import llm_service
from app.services.rag_service import rag_service
from app.schemas.extraction import ExtractionRequest, ExtractionResponse
from app.services.extraction_service import extraction_service

app = FastAPI(title=settings.PROJECT_NAME, version=settings.VERSION)

# 定義 Request Schema
class ChatRequest(BaseModel):
    message: str
    
class QueryRequest(BaseModel):
    question: str
    collection_name: str = "reference_docs" # 預設 collection

@app.get("/")
def health_check():
    return {"status": "ok", "service": "AutoFill AI"}

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
    
@app.post("/api/upload-reference")
async def upload_reference(
    files: List[UploadFile] = File(...), 
    background_tasks: BackgroundTasks = None
):
    """
    上傳參考文件(PDF/Word)，支援多檔上傳。
    使用 BackgroundTasks 在背景處理，避免使用者等待太久。
    """
    # 檢查檔案類型
    for file in files:
        if file.content_type not in ["application/pdf", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"]:
            pass 
    try:
        results = []
        for file in files:
            res = await rag_service.process_and_index_document(file)
            results.append({"filename": file.filename, "status": "success", "chunks": res["chunks"]})
        return {"uploaded_count": len(results), "details": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/query")
async def query_knowledge_base(request: QueryRequest):
    """
    RAG 問答接口：根據已上傳的文件回答問題
    """
    try:
        result = await rag_service.query_document(request.question, request.collection_name)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/api/extract", response_model=ExtractionResponse)
async def extract_form_data(request: ExtractionRequest):
    """
    自動填表 API: 接收欄位定義，回傳填完值的 JSON
    """
    try:
        results = await extraction_service.extract_fields(request.fields, request.collection_name)
        return ExtractionResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)