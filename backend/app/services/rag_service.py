import os
import shutil
import tempfile
from fastapi import UploadFile
from llama_cloud_services import LlamaParse
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # 初始化 Embedding 模型 (Text to Vector)
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                google_api_key=settings.GOOGLE_API_KEY
            )
            logger.info("Embedding service initialized.")
        except Exception as e:
            logger.error(f"Failed to initialize Embeddings: {e}")
            raise e

    async def process_and_index_document(self, file: UploadFile, collection_name: str = "reference_docs"):
        """
        上傳 -> 暫存 -> LlamaParse 解析 -> 切分 -> 向量化 -> 存入 Qdrant
        """
        temp_file_path = None
        try:
            # 將上傳的檔案存入暫存區
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_file_path = tmp.name

            logger.info(f"File saved cleanly to {temp_file_path}, starting LlamaParse...")

            # 使用 LlamaParse 解析
            # 參考: https://developers.llamaindex.ai/python/cloud/llamaparse/
            parser = LlamaParse(
                api_key=settings.LLAMA_CLOUD_API_KEY,
                result_type="markdown", 
                verbose=True,
            )
            job_result = await parser.aparse(temp_file_path)
            
            # 將 LlamaIndex 的文件格式轉換為 LangChain 的格式
            langchain_docs = [Document(page_content=page.text, metadata={"source": file.filename}) for page in job_result.pages]

            logger.info(f"Parsed {len(langchain_docs)} pages.")

            # 向量儲存到 Qdrant
            # 參考: https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant
            QdrantVectorStore.from_documents(
                documents=langchain_docs,
                embedding=self.embeddings,
                url=settings.QDRANT_URL,
                collection_name=collection_name,
                force_recreate=False # 不刪除舊資料
            )
            
            logger.info("Successfully indexed document into Qdrant.")
            return {"status": "success", "chunks": len(langchain_docs)}

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise e
        finally:
            # 清理暫存檔案 
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

rag_service = RAGService()