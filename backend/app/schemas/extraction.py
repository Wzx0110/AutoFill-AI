from pydantic import BaseModel, Field
from typing import List, Optional, Any

# 1. 單一欄位的定義 (前端告訴後端要填什麼)
class ExtractionField(BaseModel):
    key: str = Field(..., description="欄位唯一鍵值，例如 'applicant_name'")
    description: str = Field(..., description="欄位的自然語言描述，例如 '請找出申請人的全名'")
    data_type: str = Field("string", description="預期的資料類型: string, number, boolean")

# 2. 填寫請求 (整個表單)
class ExtractionRequest(BaseModel):
    collection_name: str = "reference_docs"
    fields: List[ExtractionField]

# 3. 單一欄位的答案 (後端回傳給前端)
class FieldResult(BaseModel):
    key: str
    value: Any
    source: str = Field(..., description="答案來源的文件名或頁碼")
    confidence: str = Field(..., description="AI 對答案的信心程度 (High/Medium/Low)")

# 4. 最終回傳結果
class ExtractionResponse(BaseModel):
    results: List[FieldResult]