import os
import json
import tempfile
from urllib.parse import urlparse, parse_qs

import yt_dlp
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
load_dotenv()
# Initialize LLM + Embeddings
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

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


def build_vectorstore(text: str, persist_directory: str, collection_name: str):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.create_documents([text])

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    return vectorstore


def ask_question(vectorstore, query: str) -> str:
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 5})
    docs = retriever.invoke(query)

    if not docs:
        return "NO relevant data found."

    context = "\n\n".join([d.page_content for d in docs])

    prompt = f"""
   You are a helpful assistant. Answer the question based ONLY on the provided context:
   {context}

    Follow these response formatting rules based on the keywords in the user's query:

    1. SUMMARY (keywords: "summary", "brief", "in short", "summarize"):
    - Provide a concise summary with only the most important insights.
    - Use bullet points where appropriate.
    - Avoid unnecessary details.

    2. QUESTION-ANSWER (direct questions):
    - Provide clear, direct, and complete answers.
    - Keep responses concise and relevant.

    3. ELABORATION (keywords: "explain", "elaborate", "deep dive into"):
    - Provide detailed and comprehensive explanations.
    - Include examples or data from the context if available.
    - Do not invent or add external information.

    4. ANALYSIS (keywords: "analyze", "examine", "assess", "draw insights"):
    - Provide a structured analysis that covers:
        - Current situation
        - Key factors
        - Implications
        - Conclusion

    - If the user asks what the video is about, provide a jist in about 3-4 sentences.

    STRICT INSTRUCTIONS:
    - Always answer using ONLY the provided context.
    - If no relevant information is found in the context, reply: "NO relevant data found".
    - Do not use outside knowledge or guesswork.
    - If the question is unrelated to the context, reply: "I can only answer questions related to the provided context".
    - Eliminate redundancy and keep responses precise.

    Question: {query}
    Answer:
    """
    response = llm.invoke(prompt)
    return response.content
