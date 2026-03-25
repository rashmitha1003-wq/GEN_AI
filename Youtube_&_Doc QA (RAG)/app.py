import streamlit as st
from main import get_transcript_from_url, load_document, build_vectorstore, ask_question

# persist_directory = "vectorstore_data"
# collection_name = "rag_collection_new"

st.set_page_config(page_title="YouTube/Doc RAG QA", layout="centered")
st.title("Upload and Ask")

# Sidebar input
st.sidebar.header("Upload Source")
source_type = st.sidebar.radio("Choose source type:", ["YouTube URL", "Document"])

import uuid

if source_type == "YouTube URL":
    url = st.sidebar.text_input("Enter YouTube URL:")
    if st.sidebar.button("Load YouTube Transcript") and url:
        with st.spinner("Fetching transcript..."):
            text_data = get_transcript_from_url(url)
            unique_id = uuid.uuid4().hex[:6]
            persist_directory = f"vectorstore_data/{unique_id}"
            collection_name = f"rag_collection_{unique_id}"
            st.session_state["vectorstore"] = build_vectorstore(text_data, persist_directory, collection_name)
        st.success("Transcript loaded into vectorstore!")

elif source_type == "Document":
    uploaded_file = st.sidebar.file_uploader("Upload PDF or TXT/MD file", type=["pdf", "txt", "md"])
    if uploaded_file and st.sidebar.button("Load Document"):
        with st.spinner("Processing document..."):
            text_data = load_document(uploaded_file)
            unique_id = uuid.uuid4().hex[:6]
            persist_directory = f"vectorstore_data/{unique_id}"
            collection_name = f"rag_collection_{unique_id}"
            st.session_state["vectorstore"] = build_vectorstore(text_data, persist_directory, collection_name)
        st.success("Document loaded into vectorstore!")


# QA Section
if "vectorstore" in st.session_state:
    query = st.text_input("Ask a question about your content:")
    if query:
        with st.spinner("Thinking..."):
            answer = ask_question(st.session_state["vectorstore"], query)
        st.markdown("###Answer")
        st.write(answer)
else:
    st.info("⬅️ Please load a YouTube transcript or upload a document first.")
