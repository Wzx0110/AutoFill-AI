import os
import shutil
import tempfile
import logging
from typing import List, Dict, Any

from fastapi import UploadFile
from llama_cloud_services import LlamaParse

# LangChain Core
from langchain_core.documents import Document
from langchain.agents import create_agent 
from langchain_core.messages import SystemMessage

# LangChain Google & Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Tools
from langchain_community.tools import DuckDuckGoSearchRun

from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # 初始化 Embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY
        )

        # 初始化 LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3
        )
        
        # 初始化 Tools
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = [self.search_tool]

        logger.info("LangChain Agent (LangGraph-based) Service initialized.")

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

            # 使用 LlamaParse 解析
            # 參考: https://developers.llamaindex.ai/python/cloud/llamaparse/
            parser = LlamaParse(
                api_key=settings.LLAMA_CLOUD_API_KEY, 
                result_type="markdown", 
                verbose=True
            )
            job_result = await parser.aparse(temp_file_path)
            # 將 LlamaIndex 的文件格式轉換為 LangChain 的格式
            langchain_docs = [Document(page_content=page.text, metadata={"source": file.filename}) for page in job_result.pages]

            # 向量儲存到 Qdrant
            # 參考: https://docs.langchain.com/oss/python/integrations/vectorstores/qdrant
            QdrantVectorStore.from_documents(
                documents=langchain_docs,
                embedding=self.embeddings,
                url=settings.QDRANT_URL,
                collection_name=collection_name,
                force_recreate=False # 不刪除舊資料
            )
            return {"status": "success", "chunks": len(langchain_docs), "collection": collection_name}
        except Exception as e:
            logger.error(f"Index Error: {e}")
            raise e
        finally:
            # 清理暫存檔案
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    async def query_document(self, question: str, session_id: str):
        """
        用戶提問 -> 轉向量 -> 搜尋 Qdrant -> 抓出相關文章 -> 給 LLM 整理回答
        Agentic RAG 
        1. 先去 Qdrant 撈相關文件 (Retrieval)
        2. 把文件當作 Context 塞給 Agent
        3. Agent 判斷資訊是否足夠：
           - 足夠 -> 直接回答
           - 不足 -> 呼叫 Search Tool -> 整合後回答
        """
        collection_name = f"session_{session_id}"
        
        try:
            client = QdrantClient(url=settings.QDRANT_URL)
            
            # 檢索階段 (Retrieval)
            # 參考: https://reference.langchain.com/python/integrations/langchain_qdrant/#langchain_qdrant.QdrantVectorStore
            retrieved_context = "No internal documents found."
            sources = []

            # 檢查 Collection 是否存在且有資料
            collections = client.get_collections().collections
            if any(c.name == collection_name for c in collections) and client.count(collection_name).count > 0:
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=collection_name,
                    embedding=self.embeddings,
                )
                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})
                docs = await retriever.ainvoke(question)
                
                if docs:
                    retrieved_context = "\n\n".join([d.page_content for d in docs])
                    sources = [d.metadata.get("source", "unknown") for d in docs]

            # 定義 System Prompt
            system_prompt_content = f"""
            You are a smart 'AutoFill Agent'.
            
            === INTERNAL CONTEXT (From uploaded files) ===
            {retrieved_context}
            ==============================================
            
            CRITICAL INSTRUCTIONS:
            1. First, check the INTERNAL CONTEXT. If the answer is there, use it.
            2. If the answer is NOT in the context (e.g., comparing with a competitor not in the file), you MUST use the search tool.
            3. Do not just say "I don't know". Research it.
            4. When answering, cite your sources (e.g., "According to the file..." or "Based on web search...").
            """

            # 建立 Agent
            agent = create_agent(
                model=self.llm,          
                tools=self.tools,         
                system_prompt=system_prompt_content 
            )

            # 執行 Agent
            result = await agent.ainvoke(
                {"messages": [{"role": "user", "content": question}]}
            )
            
            # 解析結果
            last_message = result["messages"][-1]
            raw_content = last_message.content
            final_answer = ""
            
            # 純字串
            if isinstance(raw_content, str):
                final_answer = raw_content  
            # 列表 
            elif isinstance(raw_content, list):
                for item in raw_content:
                    if isinstance(item, str):
                        final_answer += item
                    elif isinstance(item, dict) and "text" in item:
                        final_answer += item["text"]
            # 其他
            else:
                final_answer = str(raw_content)
            
            logger.info(f"Agent Final Answer: {final_answer[:100]}...")

            # 處理 Sources 標記 (檢查是否使用了工具)
            for msg in result["messages"]:
                # 檢查是否有 ToolMessage (代表工具被呼叫並回傳了結果)
                if msg.type == "tool": 
                     if "Internet Search" not in sources:
                        sources.append("Internet Search")

            logger.info(f"Agent Answer: {final_answer[:100]}...")

            return {
                "answer": final_answer,
                "source_documents": sources
            }

        except Exception as e:
            logger.error(f"Agent Error: {e}")
            return {
                "answer": f"Processing Error: {str(e)}",
                "source_documents": []
            }

rag_service = RAGService()