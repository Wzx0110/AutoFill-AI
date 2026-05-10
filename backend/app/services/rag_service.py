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
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

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
    search_query: str
    session_id: str
    documents: List[dict]      # 存放找到的文件片段
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
        
        self.search_tool = DuckDuckGoSearchAPIWrapper()
        
        # === 建立 LangGraph 狀態機 ===
        self.graph = self._build_graph()
        logger.info("LangGraph State Machine initialized.")

    # ==========================================
    # 定義各個節點 (Nodes) 邏輯
    # ==========================================
    
    async def rewrite_node(self, state: GraphState):
        """節點：查詢重寫 (Query Rewrite)"""
        original_question = state["question"]
        
        logger.info(f"[Rewrite Node] Original Question: {original_question}")

        # 使用 LLM 將使用者的問題轉換為最佳的向量檢索關鍵字
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Search Query Optimization Expert.
            Convert the user's conversational question into a CONCISE and highly effective search query.
            
            CRITICAL RULES:
            1. KEEP IT SHORT: Use ideally 3 to 6 words.
            2. ESSENTIALS ONLY: Extract ONLY the core entities, exact dates, and main metrics.
            3. NO FILLERS: Do NOT add descriptive filler words or synonyms like "comparison", "analysis", "performance", "what is", or "tell me".
            4. Translate to English for better search results.
            
            Return ONLY the string.
            """),
            ("human", "{question}")
        ])

        # 使用 StrOutputParser 確保只拿回純文字
        chain = prompt | self.llm | StrOutputParser()
        optimized_query = await chain.ainvoke({"question": original_question})
        
        logger.info(f"[Rewrite Node] Optimized Query: {optimized_query}")
        
        # 將重寫後的查詢存入 state
        return {"search_query": optimized_query}
    
    async def retrieve_node(self, state: GraphState):
        """節點：強制查詢本地 Qdrant 向量庫"""
        search_target = state.get("search_query", state["question"])
        session_id = state["session_id"]
        collection_name = f"session_{session_id}"
        
        docs = []
        sources = state.get("sources", set())

        try:
            client = QdrantClient(url=settings.QDRANT_URL)
            collections = client.get_collections().collections
            if any(c.name == collection_name for c in collections) and client.count(collection_name).count > 0:
                vector_store = QdrantVectorStore(client=client, collection_name=collection_name, embedding=self.embeddings)
                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})
                retrieved_docs = await retriever.ainvoke(search_target) 
                
                for d in retrieved_docs:
                    source_name = d.metadata.get("source", "unknown")
                    docs.append({
                        "source_name": source_name,
                        "url": None,
                        "content": d.page_content
                    })
                    sources.add(source_name)
                    
            logger.info(f"[Retrieve Node] Found {len(docs)} document chunks.")
        except Exception as e:
            logger.warning(f"[Retrieve Node] Error: {e}")

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
        context = "\n\n".join([doc["content"] for doc in documents])
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
        """節點：聯網搜尋 (僅在需要時觸發)"""
        search_target = state.get("search_query", state["question"])
        documents = state.get("documents", [])
        sources = state.get("sources", set())

        logger.info(f"[Web Search Node] Searching DuckDuckGo for: {search_target}")
        
        try:
            # 使用 .results() 獲取前 3 筆帶有網址的結構化資料
            search_results = self.search_tool.results(search_target, max_results=8)
            
            if search_results:
                for res in search_results:
                    title = res.get("title", "Internet Search")
                    link = res.get("link", "")
                    snippet = res.get("snippet", "")
                    
                    documents.append({
                        "source_name": title,
                        "url": link,
                        "content": snippet
                    })
                    sources.add(title)
            else:
                logger.warning("[Web Search Node] Empty results.")
                
        except Exception as e:
            logger.error(f"[Web Search Node] Search failed: {e}")

        return {"documents": documents, "sources": sources}

    async def generate_node(self, state: GraphState, config: RunnableConfig):
        """節點：結合所有收集到的資料，生成最終回答"""
        question = state["question"]
        documents = state.get("documents", [])

        grouped_docs = {}
        for doc in documents:
            key = (doc["source_name"], doc["url"])
            if key not in grouped_docs:
                grouped_docs[key] = []
            grouped_docs[key].append(doc["content"])

        context = ""
        for i, (key, contents) in enumerate(grouped_docs.items(), 1):
            source_name, url = key
            context += f"=== Source [{i}] ===\n"
            context += f"Name: {source_name}\n"
            if url:
                context += f"URL: {url}\n"
            context += "Content:\n" + "\n...\n".join(contents) + "\n\n"
            
        if not context:
            context = "No relevant context found."
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly intelligent Information Synthesis Agent.
            
            === GATHERED CONTEXT ===
            {context}
            ========================
            
            CRITICAL INSTRUCTIONS (MUST FOLLOW EXACTLY OR FAIL):
            1. DIRECT ANSWER: Start IMMEDIATELY with the facts. No filler phrases.
            2. SEQUENTIAL RENUMBERING (CRITICAL): You MUST renumber the sources you actually cite sequentially, starting from 1 (i.e., 1, 2, 3...). Do NOT skip numbers. Ignore the original Source IDs from the context.
            3. INLINE CITATIONS (MUST BE CLICKABLE): 
               - For internet sources, you MUST embed the URL inline using this EXACT format: `[[1](URL)]` (Outer brackets are plain text, the number inside is a clickable link).
               - For local documents, use plain text: `[1]`
               - NEVER combine citations. Use `[1][[2](URL)]`, NOT `[1, 2]`.
            4. SOURCES LIST FORMAT (CRITICAL): At the VERY END, add a blank line, write exactly "**Sources:**" (in bold), and list EACH cited source sequentially.
               - Use a standard numbered list (`1. `, `2. `, `3. `). Do NOT use bullet points (`-`).
               - Format for internet sources: `1. [Name](URL)`
               - Format for local documents: `1. Name`
               
               Example Footer:
               
               **Sources:**
               1. NVIDIA_FY25_Q3.pdf
               2. [AMD Earnings Report](https://example.com)
               3. [Tech News](https://news.com)
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
        workflow.add_node("rewrite", self.rewrite_node)
        workflow.add_node("retrieve", self.retrieve_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("generate", self.generate_node)
        
        # 設定流程起點 -> 先去重寫問題
        workflow.add_edge(START, "rewrite")         
        
        # 重寫完畢 -> 去檢索
        workflow.add_edge("rewrite", "retrieve")    

        # 設定條件分歧點 -> 檢索完後交給 Grader 決定下一步
        workflow.add_conditional_edges(
            "retrieve",
            self.grade_and_route,
            {
                "web_search": "web_search", # 如果回傳 web_search，走向聯網
                "generate": "generate"      # 如果回傳 generate，直接走向生成
            }
        )
        # 聯網搜尋完畢後 -> 生成
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
            # 用來收集 LLM 的完整回答
            full_generated_text = ""
            
            # 使用 config 傳遞串流設定 (給 generate_node 裡的 LLM 用)
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

            # LLM 已經把 Reference 寫在 markdown 裡了
            yield json.dumps({"type": "sources", "content": []}) + "\n"

        except Exception as e:
            logger.error(f"Stream Error: {e}", exc_info=True)
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

rag_service = RAGService()