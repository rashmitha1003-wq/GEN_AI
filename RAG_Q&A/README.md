#  QA System

This project is a Retrieval-Augmented Generation (RAG) system that allows users to ask questions about content from YouTube videos or uploaded documents. The system uses OpenAI's GPT-4 model for answering questions based on the provided context.

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

- Ensure you have a valid OpenAI API key to use the GPT-4 model.
- The system stores vectorstore data in a directory for persistence.

## License

This project is licensed under the MIT License.