import os
import json
import tempfile
from urllib.parse import urlparse, parse_qs

import yt_dlp
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
load_dotenv()

# Initialize LLM + Embeddings
llm = ChatGroq(model="openai/gpt-oss-120b")
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

# --- Utilities ---

def extract_video_id(url: str) -> str:
    query = urlparse(url)
    video_id = parse_qs(query.query).get("v")
    return video_id[0] if video_id else None


def get_transcript_from_url(youtube_url: str) -> str:
    video_id = extract_video_id(youtube_url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    with tempfile.TemporaryDirectory() as tmpdir:
        ydl_opts = {
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'subtitlesformat': 'json3',
            'skip_download': True,
            'outtmpl': f'{tmpdir}/{video_id}.%(ext)s',
            'quiet': True,
            "sleep_interval": 2,
            "max_sleep_interval": 5,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([youtube_url])

        json3_path = os.path.join(tmpdir, f"{video_id}.en.json3")
        if not os.path.exists(json3_path):
            raise FileNotFoundError("Transcript not found or not downloadable.")

        with open(json3_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        segments = data.get('events', []) 
        lines = [] 
        for segment in segments: 
            if 'segs' in segment: 
                text = ''.join([s.get('utf8', '') for s in segment['segs']]) 
                lines.append(text.strip()) 
        return ' '.join(lines)

def generate_video_insights(text: str):
    prompt = f""" 

        You are a notes assistant.
        Analyze the following video transcript and generate:

        1. SUMMARY : write a very clear summary of the transcript. Keep it concise.

        2. KEY INSIGHTS : Bullet points of the most important ideas. Make sure all the important info is covered.

        3. TOPICS COVERED : list of topics mentioned in the transcript ONLY. Do not add any extra topics.

        content:{text}
    """
    response = llm.invoke(prompt)
    return response.content

def load_document(uploaded_file) -> str:
    ext = os.path.splitext(uploaded_file.name)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmp_path = tmpfile.name

    if ext == ".pdf":
        loader = PyPDFLoader(tmp_path)
    elif ext in [".txt", ".md"]:
        loader = TextLoader(tmp_path)
    else:
        raise ValueError("Unsupported file type. Use PDF or TXT/MD.")

    docs = loader.load()
    return " ".join([d.page_content for d in docs])

from langchain_classic.schema import Document
def build_vectorstore(text, persist_directory: str, collection_name: str):
    text_splitter = SemanticChunker(embeddings = embeddings, breakpoint_threshold_type="percentile")
    docs = text_splitter.create_documents([text])

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )

    bm25 = BM25Retriever.from_documents(docs)
    bm25.k = 7

    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 7, "fetch_k": 20}
    )

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25, vector_retriever],
        weights=[0.5, 0.5]
    )

    return hybrid_retriever


def rerank_documents(query, documents, top_k=4):
    """
    query: user's question
    documents: list of LangChain Document objects
    top_k: number of reranked documents to return
    """

    if not documents:
        return []

    pairs = [
        (query, doc.page_content)
        for doc in documents
    ]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(documents, scores),
        key=lambda item: item[1],
        reverse=True
    )

    return [
        doc
        for doc, score in ranked_docs[:top_k]
    ]


# def ask_question(retriever, query: str):
#     # retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5})
#     docs = retriever.invoke(query)

#     print("\n==== RETRIEVED DOCS ====")
#     for i, d in enumerate(docs):
#         print(f"\nDoc {i+1}:")
#         print(d.page_content)

#     if not docs:
#         return {
#             "answer": "NO relevant data found.",
#             "retrieved_contexts": []
#         }

#     context = "\n\n".join([d.page_content for d in docs])
#     prompt = f"""
#    You are a helpful assistant. Answer the question based ONLY on the provided context.

# Context:
# {context}

# Question:
# {query}


#     STRICT INSTRUCTIONS:
#     - Always answer questions that are relevant to the context.
#     - Do not use outside knowledge.
#     - If no relevant information is found in the context, reply: "NO relevant data found".
#     - If the question is unrelated to the context, reply: "I can only answer questions related to the provided context".
#     - Eliminate redundancy and keep responses precise.

#     Question: {query}
#     Answer:
#     """
#     response = llm.invoke(prompt)
#     answer = response.content


#     return {
#         "answer": answer,
#         "retrieved_contexts": docs
#     }



def ask_question(retriever, query: str):
    # Step 1: Retrieve candidate chunks using your existing MMR retriever
    docs = retriever.invoke(query)

    print("\n==== ORIGINAL RETRIEVED DOCS ====")
    for i, doc in enumerate(docs, start=1):
        print(f"\nDoc {i}:")
        print(doc.page_content)

    if not docs:
        return {
            "answer": "NO relevant data found.",
            "retrieved_contexts": []
        }

    # Step 2: Rerank the retrieved chunks
    reranked_docs = rerank_documents(
        query=query,
        documents=docs,
        top_k=5
    )

    print("\n==== RERANKED DOCS ====")
    for i, doc in enumerate(reranked_docs, start=1):
        print(f"\nDoc {i}:")
        print(doc.page_content)

    # Step 3: Build context using ONLY the reranked chunks
    context = "\n\n".join(
        doc.page_content for doc in reranked_docs
    )

    # Step 4: Generate the answer
    prompt = f"""
You are a helpful assistant. Answer the question based ONLY
on the provided context.

Context:
{context}

Question:
{query}

STRICT INSTRUCTIONS:
- Answer using only information from the context.
- If the context does not contain the answer, reply:
  "NO relevant data found".
- If the question is unrelated to the provided material, reply:
  "I can only answer questions related to the provided context".
- Keep the response precise and avoid redundancy.

Answer:
"""

    response = llm.invoke(prompt)

    return {
        "answer": response.content,
        "retrieved_contexts": reranked_docs
    }



