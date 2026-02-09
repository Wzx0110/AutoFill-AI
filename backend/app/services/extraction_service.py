import logging
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from app.schemas.extraction import ExtractionField, FieldResult
from app.services.rag_service import rag_service 

logger = logging.getLogger(__name__)

class ExtractionService:
    def __init__(self):
        self.llm = rag_service.llm
        self.embeddings = rag_service.embeddings

    async def extract_fields(self, fields: List[ExtractionField], collection_name: str) -> List[FieldResult]:
        """
        核心迴圈：針對每個欄位進行 RAG 檢索與回答
        """
        results = []
        
        # 取得 Qdrant Retriever (透過 rag_service 的 helper 或重新建立)
        # 為了簡單，我們直接呼叫 rag_service 的 query_document 邏輯
        # 但為了精準控制，我們這裡稍微客製化 Prompt
        
        for field in fields:
            logger.info(f"Extracting field: {field.key} - {field.description}")
            
            try:
                # 1. 執行 RAG 檢索 (Reuse RAG Logic)
                # 我們直接呼叫之前寫好的 query_document，它已經包含了 Search + LLM Generation
                # 但這裡的 Prompt 需要微調成「精簡回答」模式
                
                # 為了避免互相影響，我們這裡直接用 rag_service.query_document 
                # 雖然它原本是設計給聊天用的 (比較囉唆)，但作為 MVP 足夠了。
                # Sprint 6 優化時，我們會寫專門的 Extraction Chain。
                
                rag_result = await rag_service.query_document(
                    question=f"請針對以下欄位需求提供精簡、準確的答案，不要有多餘的廢話：{field.description}",
                    collection_name=collection_name
                )
                
                answer = rag_result["answer"]
                sources = list(set(rag_result["source_documents"]))
                source_str = ", ".join(sources) if sources else "Unknown"

                # 2. 包裝結果
                results.append(FieldResult(
                    key=field.key,
                    value=answer,
                    source=source_str,
                    confidence="High" if sources else "Low"
                ))

            except Exception as e:
                logger.error(f"Error extracting field {field.key}: {e}")
                results.append(FieldResult(
                    key=field.key,
                    value=None,
                    source="Error",
                    confidence="None"
                ))
        
        return results

extraction_service = ExtractionService()