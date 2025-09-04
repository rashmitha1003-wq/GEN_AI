import os
import tempfile
from dotenv import load_dotenv
from typing import TypedDict, Sequence, Annotated
from sentence_transformers import CrossEncoder
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.tools import tool
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

# Load environment variables
load_dotenv()

# PDF paths (shared with Streamlit)
MEDICAL_PDF_PATH = "medical.pdf"
EXERCISE_PDF_PATH = "physical activity.pdf"

# loading, chunking and embedding the documents AND creating the vector DBs

def load_and_chunk_pdfs(medical_pdf_path, exercise_pdf_path, embeddings, logger=print):
    """Loads PDFs, chunks them semantically, and returns the chunks."""
    try:
        loader1 = PyPDFLoader(medical_pdf_path)
        medical_docs = loader1.load()

        loader2 = PyPDFLoader(exercise_pdf_path)
        exercise_docs = loader2.load()

        logger("Loaded all the documents.")

        semantic_chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")
        chunks_m = semantic_chunker.create_documents([d.page_content for d in medical_docs])
        chunks_e = semantic_chunker.create_documents([d.page_content for d in exercise_docs])
        logger("Chunking completed.")
        
        return chunks_m, chunks_e
    except Exception as e:
        logger(f"Error loading or chunking PDFs: {e}")
        return None, None

def create_retrievers(_chunks_m, _chunks_e, embeddings, temp_dir, logger=print):
    """Creates vector databases and ensemble retrievers for the documents."""
    try:
        chroma_dir = os.path.join(temp_dir, "chroma_db")
        faiss_dir = os.path.join(temp_dir, "faiss_db")
        
        chroma_db = Chroma.from_documents(
            documents=_chunks_m,
            embedding=embeddings,
            persist_directory=chroma_dir
        )
        faiss_db = FAISS.from_documents(
            documents=_chunks_e,
            embedding=embeddings
        )
        faiss_db.save_local(faiss_dir)

        #vector search
        retriever_m = chroma_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.25, "k": 6})
        retriever_e = faiss_db.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold": 0.15, "k": 6})
        #hybrid search
        bm25_retriever_m = BM25Retriever.from_documents(_chunks_m); bm25_retriever_m.k = 6
        bm25_retriever_e = BM25Retriever.from_documents(_chunks_e); bm25_retriever_e.k = 6

        ensemble_retriever_m = EnsembleRetriever(retrievers=[retriever_m, bm25_retriever_m], weights=[0.5, 0.5])
        ensemble_retriever_e = EnsembleRetriever(retrievers=[retriever_e, bm25_retriever_e], weights=[0.5, 0.5])
        
        return ensemble_retriever_m, ensemble_retriever_e
    except Exception as e:
        logger(f"Error creating retrievers: {e}")
        return None, None

def get_reranker():
    """Loads the cross-encoder model for reranking."""
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_results(query, docs, threshold, reranker):
    """Reranks retrieved documents based on a query."""
    if not docs:
        return []
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    filtered = [(doc, score) for doc, score in zip(docs, scores) if score >= threshold]
    ranked = sorted(filtered, key=lambda x: x[1], reverse=True)
    return [doc for doc, score in ranked]

#LangGraph Agent 

class AgentState(TypedDict):
    query: str
    messages: Annotated[Sequence[BaseMessage], add_messages]

def build_agent(medical_pdf=MEDICAL_PDF_PATH, exercise_pdf=EXERCISE_PDF_PATH, logger=print):
    """Builds and returns the compiled LangGraph agent."""
    temp_dir = tempfile.mkdtemp()
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    reranker = get_reranker()

    chunks_m, chunks_e = load_and_chunk_pdfs(medical_pdf, exercise_pdf, embeddings, logger)
    if not (chunks_m and chunks_e):
        return None

    ensemble_retriever_m, ensemble_retriever_e = create_retrievers(chunks_m, chunks_e, embeddings, temp_dir, logger)
    if not (ensemble_retriever_m and ensemble_retriever_e):
        return None

    #tools
    @tool
    def medical_search(query: str) -> str:
        """for medical-related queries"""
        docs = ensemble_retriever_m.invoke(query)
        if not docs:
            return "No relevant medical documents found for your query."
        reranked = rerank_results(query, docs, 0.4, reranker)
        return "\n\n".join([d.page_content for d in reranked])

    @tool
    def exercise_search(query: str) -> str:
        """for exercise-related queries"""
        docs = ensemble_retriever_e.invoke(query)
        if not docs:
            return "No relevant exercise documents found for your query."
        reranked = rerank_results(query, docs, 0.4, reranker)
        return "\n\n".join([d.page_content for d in reranked])

    tavily_search = TavilySearch(max_results=3)
    tools = [medical_search, exercise_search, tavily_search]
    llm_with_tools = llm.bind_tools(tools=tools)

    def agent(state: AgentState):
        """Agent node processing the conversation."""
        system_prompt = SystemMessage(
            content="""You are a Health Care Assistant. Please use the tools provided to answer the user's query.
                            You are provided with: 'medical_search', 'exercise_search', and 'tavily_search'.
                            **GUIDELINES**
                            1. For questions on medical diagnosis, symptoms, etc., use 'medical_search'.
                            2. For questions on physical activities and exercises, use 'exercise_search'.
                            3. If document information is incomplete, use 'tavily_search' and combine results.
                            4. Do not invent information. Keep answers user-friendly and clear."""
        )
        response = llm_with_tools.invoke([system_prompt] + state['messages'])
        return {'messages': [response]}

    def should_continue(state: AgentState):
        """Decides whether to continue tool usage based on the last message."""
        return hasattr(state['messages'][-1], 'tool_calls') and len(state['messages'][-1].tool_calls) > 0

    tool_node = ToolNode(tools=tools)
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {True: "tools", False: END})
    graph.add_edge("tools", "agent")

    return graph.compile()
