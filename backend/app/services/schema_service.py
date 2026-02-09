import logging
import tempfile
import os
import shutil
import json
from fastapi import UploadFile
from llama_cloud_services import LlamaParse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.core.config import settings
from app.schemas.extraction import ExtractionField

logger = logging.getLogger(__name__)

class SchemaService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0, # 分析欄位要精準，不要創意
            response_mime_type="application/json"
        )

    async def analyze_form(self, file: UploadFile) -> list[dict]:
        """
        1. 解析上傳的空白表格 (PDF/Word)
        2. 使用 LLM 識別所有需要填寫的欄位
        3. 回傳欄位定義列表 (JSON)
        """
        temp_file_path = None
        try:
            # 1. 儲存暫存檔
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_file_path = tmp.name

            # 2. LlamaParse 解析 (它對表格結構理解力最強)
            parser = LlamaParse(api_key=settings.LLAMA_CLOUD_API_KEY, result_type="markdown")
            documents = parser.load_data(temp_file_path)
            form_content = "\n".join([doc.text for doc in documents])
            
            # 3. LLM 分析 (Prompt Engineering)
            prompt = ChatPromptTemplate.from_template("""
            你是一個專業的資料輸入自動化專家。
            請分析以下的【表單內容】，找出所有需要使用者填寫的欄位。
            
            對於每個欄位，請輸出：
            1. "key": 英文變數名稱 (例如 applicant_name, total_revenue)
            2. "description": 這個欄位在問什麼？請轉換成一個明確的問句，方便我去搜尋答案。(例如：請找出申請人的姓名是什麼？)
            3. "data_type": string, number, boolean, or date

            【表單內容】:
            {form_content}

            請直接輸出 JSON Object，格式如下：
            {{
                "fields": [
                    {{"key": "...", "description": "...", "data_type": "..."}},
                    ...
                ]
            }}
            """)
            
            chain = prompt | self.llm | JsonOutputParser()
            
            logger.info("Analyzing form structure with LLM...")
            result = await chain.ainvoke({"form_content": form_content})
            
            return result.get("fields", [])

        except Exception as e:
            logger.error(f"Error analyzing form: {e}")
            raise e
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

schema_service = SchemaService()