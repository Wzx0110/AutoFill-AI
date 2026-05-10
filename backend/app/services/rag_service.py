import os
import shutil
import tempfile
import logging
import json
from typing import AsyncGenerator, TypedDict, Annotated, List, Set, Sequence

from fastapi import UploadFile
from llama_cloud_services import LlamaParse

# LangChain Core
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig

# LangGraph 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# LangChain Google & Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

# Tools
from langchain_community.tools import DuckDuckGoSearchRun

from app.core.config import settings

# 設定 Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================================
# 定義 Graph 狀態 (State)
# 流動在各個節點之間的資料結構
# ==========================================
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    session_id: str
    documents: List[str]      # 存放找到的文件片段
    sources: Set[str]         # 存放資料來源


class RAGService:
    def __init__(self):
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.EMBEDDING_MODEL,
            google_api_key=settings.GOOGLE_API_KEY
        )

        self.llm = ChatGoogleGenerativeAI(
            model=settings.LLM_MODEL,
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.2 # 溫度調低，更客觀
        )
        
        self.search_tool = DuckDuckGoSearchRun()
        
        # === 建立 LangGraph 狀態機 ===
        self.graph = self._build_graph()
        logger.info("LangGraph State Machine initialized.")

    # ==========================================
    # 定義各個節點 (Nodes) 邏輯
    # ==========================================
    
    async def retrieve_node(self, state: GraphState):
        """節點 1：強制查詢本地 Qdrant 向量庫"""
        question = state["question"]
        session_id = state["session_id"]
        collection_name = f"session_{session_id}"
        
        docs = []
        sources = state.get("sources", set())

        try:
            client = QdrantClient(url=settings.QDRANT_URL)
            collections = client.get_collections().collections
            if any(c.name == collection_name for c in collections) and client.count(collection_name).count > 0:
                vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=self.embeddings)
                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 3, "fetch_k": 5})
                retrieved_docs = await retriever.ainvoke(question)
                
                for d in retrieved_docs:
                    docs.append(d.page_content)
                    sources.add(d.metadata.get("source", "unknown"))
                    
            logger.info(f"📄 [Retrieve Node] Found {len(docs)} documents internally.")
        except Exception as e:
            logger.warning(f"[Retrieve Node] Error or collection not found: {e}")

        # 將找到的資料更新回 State
        return {"documents": docs, "sources": sources}

    async def grade_and_route(self, state: GraphState) -> str:
        """條件邊界 (Conditional Edge)：評估文件是否足夠回答問題"""
        question = state["question"]
        documents = state.get("documents", [])

        # 如果根本沒找到文件，直接去上網查
        if not documents:
            logger.info("[Grader] No documents found. Routing to -> Web Search.")
            return "web_search"

        # 使用 LLM 來當裁判 (Grader)
        context = "\n\n".join(documents)
        prompt = ChatPromptTemplate.from_template(
            """You are a grader assessing whether the following retrieved context is sufficient to fully answer the user's question.
            
            Context:
            {context}
            
            User Question:
            {question}
            
            If the context contains enough information to answer the question, output exactly "yes".
            If the context is missing information (e.g., asking for comparison but only one entity is found), output exactly "no".
            Do not provide any other text.
            """
        )
        
        chain = prompt | self.llm | StrOutputParser()
        score = await chain.ainvoke({"context": context, "question": question})
        score = score.strip().lower()

        if "yes" in score:
            logger.info("[Grader] Context is sufficient. Routing to -> Generate.")
            return "generate"
        else:
            logger.info("[Grader] Context is insufficient. Routing to -> Web Search.")
            return "web_search"

    async def web_search_node(self, state: GraphState):
        """節點 2：聯網搜尋 (僅在需要時觸發)"""
        question = state["question"]
        documents = state.get("documents", [])
        sources = state.get("sources", set())

        logger.info(f"[Web Search Node] Searching DuckDuckGo for: {question}")
        
        try:
            # 執行搜尋
            search_result = self.search_tool.invoke(question)
            documents.append(f"Web Search Results:\n{search_result}")
            sources.add("Internet Search")
        except Exception as e:
            logger.error(f"[Web Search Node] Search failed: {e}")

        return {"documents": documents, "sources": sources}

    async def generate_node(self, state: GraphState, config: RunnableConfig):
        """節點 3：結合所有收集到的資料，生成最終回答"""
        question = state["question"]
        documents = state.get("documents", [])

        context = "\n\n".join(documents) if documents else "No relevant context found."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a smart 'AutoFill Agent'.
            
            === GATHERED CONTEXT ===
            {context}
            ========================
            
            Instructions:
            1. Answer the user's question accurately using ONLY the gathered context above.
            2. If the context still does not contain the answer, politely state that you cannot find the information.
            3. Always cite your sources.
            """),
            ("human", "{question}")
        ])

        chain = prompt | self.llm
        
        response_message = await chain.ainvoke(
            {"context": context, "question": question},
            config=config 
        )

        logger.info("[Generate Node] Answer generated.")
        return {"messages": [response_message]}

    # ==========================================
    # 組裝 Graph (連接節點與邊界)
    # ==========================================
    def _build_graph(self):
        workflow = StateGraph(GraphState)

        # 加入所有節點
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("generate", self.generate_node)

        # 設定流程起點 -> 強制去檢索
        workflow.add_edge(START, "retrieve")

        # 設定條件分歧點 -> 檢索完後交給 Grader 決定下一步
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_and_route,
            {
                "web_search": "web_search", # 如果回傳 web_search，走向聯網
                "generate": "generate"      # 如果回傳 generate，直接走向生成
            }
        )

        # 聯網搜尋完畢後 -> 走向生成
        workflow.add_edge("web_search", "generate")
        
        # 生成完畢後 -> 結束
        workflow.add_edge("generate", END)

        return workflow.compile()

    # ==========================================
    # 對外接口
    # ==========================================
    
    async def process_and_index_document(self, file: UploadFile, session_id: str):
        collection_name = f"session_{session_id}"
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_file_path = tmp.name
            parser = LlamaParse(api_key=settings.LLAMA_CLOUD_API_KEY, result_type="markdown", verbose=True)
            job_result = await parser.aparse(temp_file_path)
            langchain_docs = [Document(page_content=page.text, metadata={"source": file.filename}) for page in job_result.pages]
            QdrantVectorStore.from_documents(documents=langchain_docs, embedding=self.embeddings, url=settings.QDRANT_URL, collection_name=collection_name, force_recreate=False)
            return {"status": "success", "chunks": len(langchain_docs), "collection": collection_name}
        except Exception as e:
            logger.error(f"Index Error: {e}")
            raise e
        finally:
            if temp_file_path and os.path.exists(temp_file_path): os.remove(temp_file_path)

    async def query_document(self, question: str, session_id: str):
        """給 extraction_service 用的非串流版本，直接呼叫 Graph"""
        inputs = {
            "question": question,
            "session_id": session_id,
            "messages": [HumanMessage(content=question)],
            "documents": [],
            "sources": set()
        }
        result = await self.graph.ainvoke(inputs)
        return {
            "answer": result["messages"][-1].content,
            "source_documents": list(result.get("sources", set()))
        }

    async def stream_query(self, question: str, session_id: str) -> AsyncGenerator[str, None]:
        """
        [Streaming] 整合自訂的 State Machine 進行串流 
        """
        inputs = {
            "question": question,
            "session_id": session_id,
            "messages": [HumanMessage(content=question)],
            "documents": [],
            "sources": set()
        }

        try:
            logger.info(f"Starting Control-Flow Stream for: {question}")
            
            # 用來在串流過程中收集來源
            collected_sources = set()
            
            # 使用 config 傳遞串流設定 (給 generate_node 裡的 LLM 用)
            from langchain_core.runnables import RunnableConfig
            config = RunnableConfig()

            async for stream_mode, chunk in self.graph.astream(inputs, stream_mode=["messages", "updates"], config=config):
                
                # A. 處理節點狀態更新 (攔截 Sources 與 Status)
                if stream_mode == "updates":
                    for node_name, node_state in chunk.items():
                        
                        # 1. 攔截 sources 
                        if "sources" in node_state:
                            collected_sources.update(node_state["sources"])

                        # 2. 攔截 web_search 狀態
                        if node_name == "web_search":
                            yield json.dumps({"type": "status", "content": "🔍 Context insufficient. Searching the web..."}) + "\n"
                
                # B. 處理文字串流 (只抓取 generate 節點產生的 LLM token)
                elif stream_mode == "messages":
                    msg, metadata = chunk
                    if metadata.get("langgraph_node") == "generate":
                        if isinstance(msg, BaseMessage) and msg.content:
                            content_str = ""
                            if isinstance(msg.content, str):
                                content_str = msg.content
                            elif isinstance(msg.content, list):
                                for item in msg.content:
                                    if isinstance(item, str): content_str += item
                                    elif isinstance(item, dict) and 'text' in item: content_str += item['text']
                            
                            if content_str:
                                yield json.dumps({"type": "token", "content": content_str}) + "\n"

            # 迴圈結束後，直接回傳剛剛收集到的 collected_sources！
            yield json.dumps({"type": "sources", "content": list(collected_sources)}) + "\n"

        except Exception as e:
            logger.error(f"Stream Error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

rag_service = RAGService()