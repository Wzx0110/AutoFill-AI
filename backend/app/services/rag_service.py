import os
import shutil
import tempfile
from fastapi import UploadFile
from llama_cloud_services import LlamaParse
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        try:
            # 初始化 Embedding 模型 (Text to Vector)
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                google_api_key=settings.GOOGLE_API_KEY
            )
            logger.info("Embedding service initialized.")
            
            # 初始化 LLM 
            self.llm = ChatGoogleGenerativeAI(
                model=settings.LLM_MODEL,
                google_api_key=settings.GOOGLE_API_KEY,
                temperature=0.3 
            )
        except Exception as e:
            logger.error(f"Failed to initialize Embeddings: {e}")
            raise e

    async def process_and_index_document(self, file: UploadFile, session_id: str):
        """
        上傳 -> 暫存 -> LlamaParse 解析 -> 切分 -> 向量化 -> 存入 Qdrant
        處理文件並存入該 Session 專屬的 Collection
        """
        # 動態生成 Collection Name
        collection_name = f"session_{session_id}"
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

            logger.info(f"Indexing to collection: {collection_name}")

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
            return {"status": "success", "chunks": len(langchain_docs), "collection": collection_name}

        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise e
        finally:
            # 清理暫存檔案 
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
    async def query_document(self, question: str, session_id: str):
        collection_name = f"session_{session_id}"
        """
        用戶提問 -> 轉向量 -> 搜尋 Qdrant -> 抓出相關文章 -> 給 LLM 整理回答
        """
        try:
            # 連接 Qdrant (Retriever)
            # 參考: https://reference.langchain.com/python/integrations/langchain_qdrant/#langchain_qdrant.QdrantVectorStore
            client = QdrantClient(url=settings.QDRANT_URL)
            # 檢查 Collection 是否存在 (避免新聊天室直接問問題報錯)
            collections = client.get_collections().collections
            has_collection = any(c.name == collection_name for c in collections)
            
            # 如果有 Collection，還要確認裡面真的有東西 (count > 0)
            has_docs = False
            if has_collection:
                count_result = client.count(collection_name=collection_name)
                has_docs = count_result.count > 0
            
            # === 分支 A: RAG 模式 (有文件) ===
            if has_docs:
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=collection_name,
                    embedding=self.embeddings,
                )
                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})

                # --- 關鍵修改：更靈活的 Prompt ---
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """
                    你是 AutoFill AI 的智慧助手。你的任務是回答使用者的問題。
                    
                    回答策略：
                    1. **優先使用**【參考文件】中的資訊來回答。
                    2. 如果使用者的問題涉及【參考文件】**以外**的知識（例如：比較競爭對手、解釋專有名詞、或是文件中缺少的常識），請使用你的**通用知識**來補充回答。
                    3. 請明確區分資訊來源。例如：「根據文件顯示 NVIDIA 的營收是...，而根據我的資料庫 AMD 的營收通常在...」。
                    4. 嚴禁瞎掰文件裡明確寫著的數字，但對於文件沒寫的，你可以提供背景知識。
                    """),
                    ("human", "【參考文件片段】:\n{context}\n\n【問題】:\n{question}"),
                ])
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
            
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | self.llm
                    | StrOutputParser()
                )
                
                response = await rag_chain.ainvoke(question)
                docs = await retriever.ainvoke(question)
                # answer (回答), source_documents (引用的原文)
                return {
                    "answer": response,
                    "source_documents": [doc.metadata.get("source", "unknown") for doc in docs]
                }

            # === 分支 B: 純 LLM 模式 (像 ChatGPT 一樣) ===
            else:
                logger.info(f"No documents found for session {session_id}. Switching to Pure LLM mode.")
                
                # 直接問 LLM，不需要 Context
                prompt = ChatPromptTemplate.from_messages([
                    ("system", "你是 AutoFill AI 的智慧助手。請根據你的通用知識回答使用者的問題。"),
                    ("human", "{question}"),
                ])
                
                chain = prompt | self.llm | StrOutputParser()
                response = await chain.ainvoke({"question": question})
                
                return {
                    "answer": response,
                    "source_documents": [] # 來源是空的
                }

        except Exception as e:
            logger.error(f"Query Error: {e}")
            raise e

rag_service = RAGService()