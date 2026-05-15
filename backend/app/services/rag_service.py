import os
import shutil
import tempfile
import logging
import asyncio
import json
from typing import AsyncGenerator, TypedDict, Annotated, List, Literal

from fastapi import UploadFile
from pydantic import BaseModel, Field
from llama_cloud_services import LlamaParse

# LangChain Core
from langchain_core.messages import BaseMessage, HumanMessage, AIMessageChunk
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

# LangGraph 
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# LangChain Google & Qdrant
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_qdrant import QdrantVectorStore, RetrievalMode, FastEmbedSparse
from qdrant_client import QdrantClient

# Tools
from tavily import AsyncTavilyClient
from app.core.config import settings 

# ------------------------------------------
# Logger Configuration (格式設定)
# ------------------------------------------
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AdaptiveRAG")

# 控制同一個 Session 的併發寫入
SESSION_LOCKS = {}

# ==========================================
# Schema 定義
# ==========================================
class RouteQuery(BaseModel):
    intent: Literal["direct_answer", "vector_search"] = Field(
        description="Choose 'direct_answer' for greetings, casual chat, or questions about your identity. Choose 'vector_search' for specific facts or requiring external context."
    )

class ContextGrade(BaseModel):
    is_relevant: bool = Field(
        description="True if the context provides useful facts for ANY PART of the query."
    )

class CompletenessCheck(BaseModel):
    is_complete: bool = Field(
        description="True if the provided documents can FULLY answer the user's query."
    )
    missing_query: str = Field(
        description="If is_complete is False, provide a targeted search query to find the missing information on the web. If True, leave empty."
    )

# ==========================================
# Graph 狀態定義
# ==========================================
class GraphState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    question: str
    session_id: str
    documents: List[dict]           
    web_search_needed: bool         
    web_search_query: str           

# ==========================================
# Adaptive RAG Service
# ==========================================
class RAGService:
    def __init__(self):
        logger.info("[System] Initializing RAGService components...")
        self.embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL, google_api_key=settings.GOOGLE_API_KEY)
        self.llm = ChatGoogleGenerativeAI(model=settings.LLM_MODEL, google_api_key=settings.GOOGLE_API_KEY, temperature=0.1, streaming=True)
        self.tavily_client = AsyncTavilyClient(api_key=settings.TAVILY_API_KEY)
        
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)
        self.sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
        self.cross_encoder_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        
        self.graph = self._build_graph()
        logger.info("[System] RAGService initialization completed successfully.")

    # ------------------------------------------
    # 意圖路由 (Query Router)
    # ------------------------------------------
    async def analyze_intent(self, state: GraphState) -> str:
        session_id = state.get("session_id", "unknown")
        logger.info(f"[Router] Analyzing query intent. (Session: {session_id})")
        
        prompt = ChatPromptTemplate.from_template(
            "Route the user's query to either 'direct_answer' (greetings, general knowledge) or 'vector_search' (requires specific data/facts).\nQuestion: {question}"
        )
        chain = prompt | self.llm.with_structured_output(RouteQuery)
        try:
            res = await chain.ainvoke({"question": state["question"]})
            logger.info(f"[Router] Intent resolved: '{res.intent}' (Session: {session_id})")
            return res.intent
        except Exception as e:
            logger.warning(f"[Router] Intent parsing failed: {str(e)}. Defaulting to 'vector_search'. (Session: {session_id})")
            return "vector_search"

    async def direct_answer_node(self, state: GraphState, config: RunnableConfig):
        session_id = state.get("session_id", "unknown")
        logger.info(f"[Node: DirectAnswer] Executing direct response. (Session: {session_id})")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Answer directly and naturally."),
            ("human", "{question}")
        ])
        response = await (prompt | self.llm).ainvoke({"question": state["question"]}, config=config)
        return {"messages": [response]}

    # ------------------------------------------
    # 檢索、相關性評估與完整性檢查 (Retrieval & Grader)
    # ------------------------------------------
    async def retrieve_and_grade_node(self, state: GraphState):
        session_id = state.get("session_id", "unknown")
        question = state["question"]
        collection_name = f"session_{session_id}"
        
        logger.info(f"[Node: Retriever] Initiating hybrid retrieval pipeline. (Session: {session_id})")
        
        collections = [c.name for c in self.qdrant_client.get_collections().collections]
        if collection_name not in collections:
            logger.warning(f"[Retriever] Collection '{collection_name}' not found. Flagging for Web Search fallback. (Session: {session_id})")
            return {"documents":[], "web_search_needed": True, "web_search_query": question}

        # 混合檢索 (Hybrid Search)
        retriever = QdrantVectorStore(
            client=self.qdrant_client, collection_name=collection_name, 
            embedding=self.embeddings, sparse_embedding=self.sparse_embeddings,
            retrieval_mode=RetrievalMode.HYBRID
        ).as_retriever(search_kwargs={"k": 10})
        
        fetched_docs = await retriever.ainvoke(question)
        logger.info(f"[Retriever] Fetched {len(fetched_docs)} raw documents from vector store. (Session: {session_id})")
        
        if not fetched_docs:
            logger.info(f"[Retriever] No documents retrieved. Flagging for Web Search fallback. (Session: {session_id})")
            return {"documents":[], "web_search_needed": True, "web_search_query": question}

        # 重排序 (Reranker)
        pairs = [[question, doc.page_content] for doc in fetched_docs]
        scores = await asyncio.to_thread(self.cross_encoder_model.score, pairs)
        candidates = [doc for doc, score in zip(fetched_docs, scores) if float(score) > 0]
        logger.info(f"[Reranker] {len(candidates)} documents passed the cross-encoder threshold. (Session: {session_id})")

        # 文件評分器
        async def grade_doc(doc):
            prompt = ChatPromptTemplate.from_template(
    """You are a highly strict document evaluator. 
    Your task is to determine if the provided context contains specific, useful information that directly answers ANY PART of the user's query.
    
    CRITICAL EVALUATION RULES:
    1. PARTIAL OR FULL ENTITY MATCH: The context MUST explicitly discuss AT LEAST ONE of the core entities or subjects mentioned in the query. 
       **IMPORTANT**: If the user's query involves multiple entities, subjects, or a comparison between them, a document that contains substantive facts about JUST ONE of those entities is STILL HIGHLY RELEVANT. You must answer True in this case.
    2. NO KEYWORD ILLUSION: Do NOT answer True just because the context shares generic keywords (e.g., dates, years, locations, industry terms) but discusses a completely different subject.
    3. SUBSTANTIVE CONTENT: If the core entity is only mentioned in passing (e.g., in a list of partners or a disclaimer) without providing relevant facts to answer the query, answer False.
    
    Return True ONLY if the context is demonstrably useful for answering at least a part of the query. Otherwise, return False.
    
    Query: {question}
    Context: {context}"""
)
            try:
                res = await (prompt | self.llm.with_structured_output(ContextGrade)).ainvoke({
                    "question": question, "context": doc.page_content
                })
                return doc if res.is_relevant else None
            except Exception as e:
                logger.debug(f"[Grader] Error grading document: {e}")
                return None 

        results = await asyncio.gather(*(grade_doc(d) for d in candidates[:5]))
        valid_docs = [{"content": d.page_content, "source": d.metadata.get("source", "Local Document")} for d in results if d]
        logger.info(f"[Grader] {len(valid_docs)} documents evaluated as highly relevant. (Session: {session_id})")

        # 完整性檢查器 (Completeness Check)
        if not valid_docs:
            logger.info(f"[Grader] Zero relevant local documents found. Triggering Web Search. (Session: {session_id})")
            return {"documents":[], "web_search_needed": True, "web_search_query": question}

        context_str = "\n".join([d["content"] for d in valid_docs])
        prompt_complete = ChatPromptTemplate.from_template("""
        You are an AI data completeness checker.
        Your task is to determine if the provided context FULLY answers every aspect of the user's query.
        
        CRITICAL RULE FOR MULTI-SUBJECT QUERIES:
        If the query asks about multiple distinct subjects, entities, or concepts, the context MUST provide sufficient information for ALL of them to be considered complete. If the context only covers a subset of the requested subjects (e.g., providing data for Entity A but missing Entity B), then it is NOT complete, and you must generate a missing query to search for the absent information.
        
        Query: {question}
        Context: {context}
        """)
        
        try:
            check_res = await (prompt_complete | self.llm.with_structured_output(CompletenessCheck)).ainvoke({
                "question": question, "context": context_str
            })
            needs_web = not check_res.is_complete
            web_query = check_res.missing_query if needs_web else question
            logger.info(f"[Grader] Completeness check resolved. is_complete={check_res.is_complete}. (Session: {session_id})")
        except Exception as e:
            logger.warning(f"[Grader] Completeness check failed: {e}. Defaulting to no web search. (Session: {session_id})")
            needs_web = False
            web_query = question

        if needs_web:
            logger.info(f"[Grader] Formulated missing information query for Web Search: '{web_query}' (Session: {session_id})")

        return {
            "documents": valid_docs, 
            "web_search_needed": needs_web,
            "web_search_query": web_query
        }

    def route_to_web_search_or_generate(self, state: GraphState) -> str:
        """根據狀態決定是否觸發網路搜尋"""
        decision = "web_search" if state.get("web_search_needed") else "generate"
        logger.info(f"[Graph Router] Node routing decision: '{decision}'. (Session: {state.get('session_id')})")
        return decision

    # ------------------------------------------
    # 網路搜尋 (Web Search)
    # ------------------------------------------
    async def web_search_node(self, state: GraphState):
        session_id = state.get("session_id", "unknown")
        search_query = state.get("web_search_query", state["question"])
        docs = state.get("documents", []) 
        
        logger.info(f"[Node: WebSearch] Executing external search for query: '{search_query}'. (Session: {session_id})")
        
        try:
            search_results = await self.tavily_client.search(query=search_query, max_results=3)
            found_results = search_results.get("results", [])
            logger.info(f"[WebSearch] Successfully retrieved {len(found_results)} web results. (Session: {session_id})")
            
            for res in found_results:
                docs.append({
                    "content": res.get("content", ""),
                    "source": res.get("url", "Web Search")
                })
        except Exception as e:
            logger.error(f"[WebSearch] Tavily search API failed: {e}. Proceeding with existing local context.", exc_info=True)
            
        return {"documents": docs}

    # ------------------------------------------
    # 生成與防幻覺 (Generate & Anti-Hallucination)
    # ------------------------------------------
    async def generate_node(self, state: GraphState, config: RunnableConfig):
        session_id = state.get("session_id", "unknown")
        docs = state.get("documents", [])
        logger.info(f"[Node: Generator] Synthesizing final response utilizing {len(docs)} document chunks. (Session: {session_id})")
        
        # 建立獨立的來源 ID 映射表
        source_map = {}
        current_id = 1
        for d in docs:
            src = d['source']
            if src not in source_map:
                source_map[src] = current_id
                current_id += 1
                
        # 建立帶有 [ID] 標記的 Context 字串
        context_parts = []
        for d in docs:
            src_id = source_map[d['source']]
            context_parts.append(f"[{src_id}] Source: {d['source']}\nContent: {d['content']}")
        context_str = "\n\n".join(context_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly intelligent and structured Assistant. 
            
            === GATHERED EVIDENCE ===
            {context}
            =========================
            
            CRITICAL INSTRUCTIONS (FORMAT & ANTI-HALLUCINATION):
            1. Answer the question using ONLY the provided evidence. If you don't know, say "I cannot answer this based on the provided sources."
            2. STRUCTURE: Use Markdown formatting. Use headers (###) and bullet points to organize the information clearly. 
                - If comparing multiple subjects, create a section for each subject.
                - End with a '綜合比較總結' (Summary) section if applicable.
            3. INLINE CITATIONS & DYNAMIC RENUMBERING (CRITICAL): 
                - You MUST append citations at the end of every sentence or bullet point that uses evidence.
                - **RENUMBERING RULE:** You must dynamically renumber the citations in your final output so they are strictly sequential (i.e., `[1]`, `[2]`, `[3]`, `[4]`). Do NOT skip numbers, even if the original [ID] in the Gathered Evidence skipped.
                - For internet sources, you MUST use valid Markdown link syntax where the citation number is the clickable text. Format: `[[1]](URL)` (e.g., `[[2]](https://example.com)`).
                - For local documents, use plain text: `[1]`.
            4. FOOTER (SOURCES LIST):
                - At the very end of your response, add a blank line, type "**Sources:**".
                - DO NOT write them on a single line.
                - Example Format:
                **Sources:**
                1. EXAMPLE.pdf
                2. [Example Web Source](https://example.com)
            """),
            ("human", "{question}")
        ])

        response = await (prompt | self.llm).ainvoke(
            {"context": context_str, "question": state["question"]}, config=config
        )
        logger.info(f"[Generator] Response synthesis completed successfully. (Session: {session_id})")
        return {"messages": [response]}

    # ------------------------------------------
    # LangGraph 狀態機組裝
    # ------------------------------------------
    def _build_graph(self):
        workflow = StateGraph(GraphState)
        
        workflow.add_node("direct_answer", self.direct_answer_node)
        workflow.add_node("retrieve_and_grade", self.retrieve_and_grade_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("generate", self.generate_node)
        
        # 起點路由 (Router)
        workflow.add_conditional_edges(
            START, 
            self.analyze_intent, 
            {"direct_answer": "direct_answer", "vector_search": "retrieve_and_grade"}
        )
        workflow.add_edge("direct_answer", END)

        # 相關性評估與分支 (Grader -> Web Search / Generate)
        workflow.add_conditional_edges(
            "retrieve_and_grade", 
            self.route_to_web_search_or_generate, 
            {"web_search": "web_search", "generate": "generate"}
        )
        
        workflow.add_edge("web_search", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()

    # ------------------------------------------
    # 入向量資料庫 Pipeline (解析 -> 切塊 -> 向量化)
    # ------------------------------------------
    async def process_and_index_document(self, file: UploadFile, session_id: str):
        collection_name = f"session_{session_id}"
        temp_path = None
        
        logger.info(f"[Indexer] Starting ingestion pipeline for file: '{file.filename}'. (Session: {session_id})")
        
        if session_id not in SESSION_LOCKS:
            SESSION_LOCKS[session_id] = asyncio.Lock()
            
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_path = tmp.name
                
            logger.info(f"[Indexer] Extracted file to temporary path. Initiating LlamaParse... (Session: {session_id})")
            
            # LlamaParse 解析
            parser = LlamaParse(api_key=settings.LLAMA_CLOUD_API_KEY, result_type="markdown")
            job_result = await parser.aparse(temp_path)
            full_md_text = "\n\n".join([p.text for p in job_result.pages])
            logger.info(f"[Indexer] LlamaParse successfully parsed {len(job_result.pages)} pages. (Session: {session_id})")

            # 語意切塊
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")])
            md_docs = md_splitter.split_text(full_md_text)
            final_chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150).split_documents(md_docs)

            for chunk in final_chunks:
                chunk.metadata["source"] = file.filename
                
            logger.info(f"[Indexer] Text splitting complete. Generated {len(final_chunks)} distinct chunks. (Session: {session_id})")

            # 確保同一個 Session 一次只有一個檔案在處理 Qdrant 操作
            async with SESSION_LOCKS[session_id]:
                logger.info(f"[Indexer] Acquired Qdrant lock for ingestion. (Session: {session_id})")
                collection_exists = self.qdrant_client.collection_exists(collection_name)

                if not collection_exists:
                    logger.info(f"[Indexer] Collection '{collection_name}' not found. Initializing new collection and inserting vectors...")
                    await QdrantVectorStore.afrom_documents(
                        documents=final_chunks,
                        embedding=self.embeddings,
                        sparse_embedding=self.sparse_embeddings,
                        collection_name=collection_name,
                        url=settings.QDRANT_URL, 
                        retrieval_mode=RetrievalMode.HYBRID
                    )
                else:
                    logger.info(f"[Indexer] Collection '{collection_name}' exists. Appending new vectors...")
                    store = QdrantVectorStore(
                        client=self.qdrant_client,
                        collection_name=collection_name,
                        embedding=self.embeddings, 
                        sparse_embedding=self.sparse_embeddings,
                        retrieval_mode=RetrievalMode.HYBRID
                    )
                    await store.aadd_documents(final_chunks)

            logger.info(f"[Indexer] Ingestion pipeline successfully completed for '{file.filename}'. (Session: {session_id})")
            return {"status": "success", "chunks_indexed": len(final_chunks)}
            
        except Exception as e:
            logger.error(f"[Indexer] Fatal error during document processing for '{file.filename}': {str(e)}", exc_info=True)
            raise e
        finally:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.debug(f"[Indexer] Cleanup successful. Removed temporary file: {temp_path}")
                except Exception as cleanup_error:
                    logger.warning(f"[Indexer] Cleanup failed for temporary file {temp_path}: {cleanup_error}")

    # ------------------------------------------
    # API 呼叫：流式輸出
    # ------------------------------------------
    async def stream_query(self, question: str, session_id: str) -> AsyncGenerator[str, None]:
        logger.info(f"[Stream API] Initializing query stream. (Session: {session_id}) | Question: '{question}'")
        
        inputs = {
            "question": question, 
            "session_id": session_id,
            "messages": [HumanMessage(content=question)],
            "documents": [], 
            "web_search_needed": False,
            "web_search_query": ""
        }

        try:
            async for mode, chunk in self.graph.astream(inputs, stream_mode=["messages", "updates"], config=RunnableConfig()):
                if mode == "updates":
                    for n, state in chunk.items():
                        if n == "retrieve_and_grade" and state.get("web_search_needed"):
                            search_target = state.get("web_search_query", question)
                            yield json.dumps({"type": "status", "content": f"觸發網路搜尋以補齊資訊：{search_target}..."}) + "\n"
                
                elif mode == "messages":
                    msg, meta = chunk
                    if meta.get("langgraph_node") in ["generate", "direct_answer"]:
                        if isinstance(msg, AIMessageChunk) and msg.content:
                            content_str = msg.content if isinstance(msg.content, str) else "".join([i.get("text", "") for i in msg.content])
                            if content_str:
                                yield json.dumps({"type": "token", "content": content_str}) + "\n"
                                
            logger.info(f"[Stream API] Query stream successfully concluded. (Session: {session_id})")
            
        except Exception as e:
            logger.error(f"[Stream API] Encountered exception during streaming: {str(e)} (Session: {session_id})", exc_info=True)
            yield json.dumps({"type": "error", "content": str(e)}) + "\n"

# 初始化實體
rag_service = RAGService()