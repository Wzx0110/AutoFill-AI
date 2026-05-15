import logging
import tempfile
import os
import shutil
from fastapi import UploadFile
from llama_cloud_services import LlamaParse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from app.core.config import settings

logger = logging.getLogger(__name__)

class SchemaService:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0, # 分析欄位要精準，不要創意
            response_mime_type="application/json"
        )

    async def analyze_form(self, file: UploadFile, global_context: str = "") -> list[dict]:
        """
        1. 解析上傳的空白表格 (PDF/Word)
        2. 根據可選的 Global Context 調整 Prompt
        3. 使用 LLM 識別所有需要填寫的欄位，並將 Context 縫合進問句中
        """
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_file_path = tmp.name

            # LlamaParse 解析
            parser = LlamaParse(api_key=settings.LLAMA_CLOUD_API_KEY, result_type="markdown")
            documents = await parser.aload_data(temp_file_path) 
            form_content = "\n".join([doc.text for doc in documents])
            
            # 動態組合 Context 指令
            context_instruction = ""
            if global_context.strip():
                logger.info(f"Using Global Context: '{global_context}'")
                context_instruction = f"""
                [Global Context Provided]:
                The user has provided the following background context: "{global_context}"
                
                CRITICAL RULE: You MUST seamlessly integrate this background context into the `description` question for EVERY field. Make sure each description is a self-contained, standalone question.
                """
            else:
                logger.info("No Global Context provided. Relying purely on form content.")
                context_instruction = """
                [NOTE - No Global Context Provided]: 
                Please infer the appropriate subject, entity, or time frame directly from the form's title or content.
                If the form lacks clear contextual clues, formulate the question generically based on the field name. 
                """

            prompt = ChatPromptTemplate.from_template("""
            You are an expert in automated form understanding and structured data extraction.

            Analyze the following form content and identify all user-fillable fields.

            {context_instruction}

            FIELD EXTRACTION RULES:
            - Extract only actual input fields requiring user-provided values
            - Do NOT extract instructions, examples, headers, or static text
            - Infer field meaning using:
            1. Nearby labels
            2. Section headers
            3. Document title
            4. Global Context (if needed)

            For each field output:

            1. "key"
            - concise snake_case variable name
            - descriptive but not overly long

            2. "description"
            - a grammatically complete standalone question.
            - understandable WITHOUT viewing the form or any other questions.
            - CRITICAL: You MUST include the core entity/subject (e.g., specific Company Name, Person Name, Year) in EVERY SINGLE description. Do NOT omit the subject to avoid repetition. It is mandatory to repeat the global context in every question.

            3. "data_type"
            Must be exactly one of:
            - "string"
            - "number"
            - "boolean"
            - "date"

            Type Rules:
            - Use "date" for calendar dates
            - Use "number" for amounts, counts, percentages, currency
            - Use "boolean" for checkboxes or yes/no fields
            - Otherwise use "string"

            [Form Content]:
            {form_content}

            Output ONLY valid JSON:

            {{
            "fields": [
                {{
                "key": "...",
                "description": "...",
                "data_type": "..."
                }}
            ]
            }}
            """)
            
            chain = prompt | self.llm | JsonOutputParser()
            
            logger.info("Analyzing form structure with LLM...")
            result = await chain.ainvoke({
                "context_instruction": context_instruction,
                "form_content": form_content
            })
            
            return result.get("fields", [])

        except Exception as e:
            logger.error(f"Error analyzing form: {e}")
            raise e
        finally:
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

schema_service = SchemaService()