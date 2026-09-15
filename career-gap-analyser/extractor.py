from langchain_community.document_loaders import UnstructuredWordDocumentLoader, PyPDFLoader 
def extract_resume_text(file):     
    """Extracts text from a resume file."""     
    if file.endswith("pdf"):         
        loader = PyPDFLoader(file)     
    else:         
        loader = UnstructuredWordDocumentLoader(file)     
    docs = loader.load()
    return "\n".join([d.page_content for d in docs])