# Health Assistant

Health Assistant is a Streamlit-based application that leverages LangChain and OpenAI technologies to provide health-related assistance. It uses local PDFs and external tools to answer medical and exercise-related queries.

## Features

- **Medical Search**: Retrieve information from medical documents.
- **Exercise Search**: Retrieve information from exercise-related documents.
- **Tavily Search**: Perform external searches for additional information.
- **Interactive UI**: Built with Streamlit for an easy-to-use interface.

## Prerequisites

- Python 3.10 or higher
- OpenAI API Key (set in a `.env` file)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Agentic-RAG-2
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Add your OpenAI API key to a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   ```

4. Ensure the required PDF files (`medical.pdf` and `physical activity.pdf`) are in the root directory.

## Usage

1. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

2. Open the app in your browser and click "Initialize Agent" to set up the system.

3. Ask health-related questions in the provided input field.

## Project Structure

- `app.py`: Streamlit application for user interaction.
- `main.py`: Core logic for building the agent and processing queries.
- `requirements.txt`: List of Python dependencies.
- `medical.pdf` and `physical activity.pdf`: Local PDF files used for document retrieval.
- `Chroma-collections/` and `faiss-collections/`: Directories for storing vector databases.

## Technologies Used

- **Streamlit**: For building the user interface.
- **LangChain**: For document processing and retrieval.
- **OpenAI**: For language model and embeddings.
- **FAISS**: For vector similarity search.
- **Chroma**: For document storage and retrieval.

## License

This project is licensed under the MIT License. See the LICENSE file for details.