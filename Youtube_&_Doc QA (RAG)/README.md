#  QA System

This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about content from YouTube videos or uploaded documents. 

Ran an evaluation pipeline and improved the quality accordingly

## Features

- **YouTube Transcript QA**: Extracts transcripts from YouTube videos and builds a vectorstore for question answering.
- **Document QA**: Supports PDF, TXT, and Markdown files for question answering.
- **Streamlit Interface**: Provides an easy-to-use web interface for interacting with the system.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/rashmitha1003-wq/GEN_AI.git
   cd basic-rag
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Create a `.env` file in the root directory.
   - Add your OpenAI API key:
     ```
     OPENAI_API_KEY=your_openai_api_key
     ```

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser (usually at `http://localhost:8501`).

3. Choose a source type from the sidebar:
   - **YouTube URL**: Enter a YouTube video URL to fetch its transcript.
   - **Document**: Upload a PDF, TXT, or Markdown file.

4. Ask questions about the loaded content in the QA section.

## Project Structure

- `app.py`: Streamlit app for the user interface.
- `main.py`: Core logic for transcript extraction, document loading, vectorstore building, and question answering.
- `requirements.txt`: List of required Python packages.
- `vectorstore_data/`: Directory for storing vectorstore data.

## Dependencies

The project uses the following Python libraries:

- `python-dotenv`
- `langchain`
- `openai`
- `chromadb`
- `youtube-transcript-api`
- `yt-dlp`
- `pypdf`
- `streamlit`

## Notes

- Ensure you have a valid Groq API key.
- The system stores vectorstore data in a directory for persistence.

## RAG Evaluation

- I evaluated the retrieval and generation quality of the RAG pipeline using RAGAS, and Langsmith for tracing and monitoring.

Evaluation setup:
 - dataset : 10 examples
 - framework: RAGAS
 - LLM: Groq openai/gpt-oss-120b
 - Embeddings: sentence-transformers/all-MiniLM-L6-v2

Final results: 
| Metric | Score |
|---|---:|
| Context Precision | 0.7478 |
| Context Recall | 0.9000 |
| Faithfulness | 0.9267 |
| Answer Relevancy | 0.7518 |

initial score for retrieval: context precision was 0.5012 an context recall was 0.7042
added hybrid retrieval and re-ranker to improve scoring.

Limitations: 
The eval uses a small manually curated testset of 10 questions. Further testing is needed to assess generalization and edge cases

## License

This project is licensed under the MIT License.