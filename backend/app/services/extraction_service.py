import logging
from typing import List
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.schemas.extraction import ExtractionField, FieldResult
from app.services.rag_service import rag_service

logger = logging.getLogger(__name__)

class ExtractionService:
    def __init__(self):
        self.llm = rag_service.llm
        self.embeddings = rag_service.embeddings
        self.search_tool = DuckDuckGoSearchRun() 
        
    async def extract_fields(self, fields: List[ExtractionField], session_id: str) -> List[FieldResult]:
        results = []
        
        for field in fields:
            logger.info(f"Extracting: {field.key}")
            
            # 根據資料類型給予不同指令
            type_instruction = ""
            if field.data_type == "number":
                type_instruction = "Output ONLY the number. No currency symbols ($, ¥), no commas, no text units. Example: 500000 not $500k."
            elif field.data_type == "date":
                type_instruction = "Output ONLY the date in YYYY-MM-DD format."
            elif field.data_type == "boolean":
                type_instruction = "Output ONLY 'True' or 'False'."
            else:
                type_instruction = "Output ONLY the exact value string. Do not use full sentences."

            # 組合 Prompt
            query_prompt = f"""
            Task: Extract the value for the field described as: "{field.description}"
            
            Rules:
            1. {type_instruction}
            2. Do NOT explain your answer.
            3. Do NOT mention "According to the document".
            4. If the information is not found, output exactly "MISSING".
            """

            # === 階段 1: 內部文件 RAG ===
            try:
                rag_answer_pack = await rag_service.query_document(
                    question=f"{field.description} (If you don't know the specific value, please answer MISSING)",
                    session_id=session_id
                )
                
                answer = rag_answer_pack["answer"]
                sources = rag_answer_pack["source_documents"]
                
                # 簡單的後處理 (Post-processing)
                if field.data_type == "number":
                    # 移除可能的非數字字符 (保留小數點和負號)
                    import re
                    # 簡單過濾，若 AI 還是回話，這裡會盡量救回來
                    numeric_match = re.search(r'-?\d*\.?\d+', answer.replace(',', ''))
                    if numeric_match:
                        answer = numeric_match.group()

                # 判斷是否需要聯網
                # 如果回答包含 "MISSING" 或 "未提及" 或來源是空的
                # 關鍵修改：判斷是否需要聯網
                # 條件 1: 答案包含 "MISSING" (LLM 承認不知道)
                # 條件 2: 答案包含 "未提及"
                # 條件 3: 完全沒有來源文件 (表示是純 LLM 瞎猜，對於精確填表來說不可靠，除非是常識題)
                # 但如果是「填表」，通常我們不希望它用通用知識瞎掰數據，所以 "not sources" 應該觸發聯網
                
                needs_web_search = "MISSING" in answer or "未提及" in answer or not sources
                
                # 特例：如果欄位是 Date/Boolean 這種通用問題，純 LLM 可能答對，但在填表場景下
                # 我們假設使用者通常是問「這家公司的營收」，所以沒文件通常就該去搜尋。
                
                if not needs_web_search:
                    results.append(FieldResult(
                        key=field.key, value=answer, source=str(sources), confidence="High (Doc)"
                    ))
                    continue 

            except Exception as e:
                logger.warning(f"RAG/LLM failed for {field.key}, trying web search...")

            # === 階段 2: 網路搜尋 (Web Search Fallback) ===
            try:
                logger.info(f"Searching web for: {field.description}")
                # 搜尋網路
                search_results = self.search_tool.invoke(field.description)
                
                # 讓 LLM 根據搜尋結果整理答案
                summary_prompt = ChatPromptTemplate.from_template("""
                請根據以下的網路搜尋結果，回答問題：{question}
                
                【搜尋結果】：
                {context}
                
                請直接給出答案值，不要有多餘的廢話。若還是找不到，請回答 "N/A"。
                """)
                
                chain = summary_prompt | self.llm | StrOutputParser()
                web_answer = await chain.ainvoke({"question": field.description, "context": search_results})
                
                results.append(FieldResult(
                    key=field.key, value=web_answer, source="Internet Search", confidence="Medium (Web)"
                ))

            except Exception as e:
                logger.error(f"Web search failed for {field.key}: {e}")
                results.append(FieldResult(key=field.key, value="N/A", source="None", confidence="None"))
        
        return results

extraction_service = ExtractionService()