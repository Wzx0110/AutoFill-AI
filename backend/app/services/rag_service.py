import os
import shutil
import tempfile
import logging
from typing import List

from fastapi import UploadFile
from llama_cloud_services import LlamaParse

# LangChain Core
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# LangChain Google & Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
# Tools & Agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from app.core.config import settings

logger = logging.getLogger(__name__)

class RAGService:
    def __init__(self):
        # 初始化 Embeddings
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=settings.EMBEDDING_MODEL,
                google_api_key=settings.GOOGLE_API_KEY
            )
        except Exception as e:
            logger.error(f"Embeddings Init Error: {e}")
            raise e

        # 初始化 LLM 
        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3
        )
        
        # 初始化 Tools
        self.search_tool = DuckDuckGoSearchRun()
        self.tools = [self.search_tool]

        logger.info("Agentic RAG Service initialized.")

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
                verbose=True,
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
            logger.error(f"Error processing document: {e}")
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
            # 連接 Qdrant (Retriever)
            # 參考: https://reference.langchain.com/python/integrations/langchain_qdrant/#langchain_qdrant.QdrantVectorStore
            client = QdrantClient(url=settings.QDRANT_URL)
            
            # 檢查是否有文件 (Retrieval Stage)
            collections = client.get_collections().collections
            has_collection = any(c.name == collection_name for c in collections)
            retrieved_context = ""
            sources = []

            if has_collection and client.count(collection_name).count > 0:
                vector_store = QdrantVectorStore(
                    client=client,
                    collection_name=collection_name,
                    embedding=self.embeddings,
                )
                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})
                
                # 執行檢索
                docs = await retriever.ainvoke(question)
                # 整理檢索到的內容
                retrieved_context = "\n\n".join([d.page_content for d in docs])
                sources = [d.metadata.get("source", "unknown") for d in docs]
            else:
                retrieved_context = "No internal documents uploaded for this session."
                
            # 定義 Agent 的 Prompt 
            prompt = ChatPromptTemplate.from_messages([
                ("system", """
                You are a smart AI assistant named 'AutoFill Agent'.
                
                You have access to the following INTERNAL CONTEXT from user uploaded files:
                <internal_context>
                {context}
                </internal_context>

                Your Goal: Answer the user's question accurately.
                
                Strategy:
                1. FIRST, check the <internal_context>. If the answer is there, use it.
                2. IF (and only if) the internal context is missing information (e.g., about competitors, latest news, or specific facts not in the file), USE YOUR SEARCH TOOL.
                3. When answering, cite your sources. Say "According to the document..." or "Based on web search...".
                """),
                ("human", "{input}"),
                ("placeholder", "{agent_scratchpad}"), # Agent 思考和呼叫工具的記憶區
            ])

            # 建立 Agent
            agent = create_tool_calling_agent(self.llm, self.tools, prompt)
            
            # 建立執行器 (AgentExecutor)
            agent_executor = AgentExecutor(
                agent=agent, 
                tools=self.tools, 
                verbose=True, 
                handle_parsing_errors=True
            )

            # 執行
            response = await agent_executor.ainvoke({
                "input": question,
                "context": retrieved_context
            })

            # 回傳結果
            raw_output = response.get("output", "")
            final_answer = ""
            
            # 模型回傳標準字串 (String)
            if isinstance(raw_output, str):
                final_answer = raw_output
            
            # 模型回傳結構化列表 (List of Dicts) 
            elif isinstance(raw_output, list):
                for item in raw_output:
                    if isinstance(item, dict) and "text" in item:
                        final_answer += item["text"]
                    elif isinstance(item, str):
                        final_answer += item
            
            # 情況 C: 其他未知格式，轉字串保命
            else:
                final_answer = str(raw_output)
            
            logger.info(f"Agent Final Answer: {final_answer[:100]}...")

            # 用到 search tool，標記 source
            if "duckduckgo" in str(response).lower() or "search" in final_answer.lower():
                if "Internet Search" not in sources:
                    sources.append("Internet Search")

            return {
                "answer": final_answer,
                "source_documents": sources
            }

        except Exception as e:
            logger.error(f"Agent Query Error: {e}")
            return {
                "answer": f"I encountered an error while thinking: {str(e)}",
                "source_documents": []
            }

rag_service = RAGService()