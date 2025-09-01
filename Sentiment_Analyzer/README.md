# Sentiment Analyzer

## Description
This project is a sentiment analysis application that uses a fine-tuned BERT model to classify text into three sentiment categories: positive, neutral, and negative. The project includes training the model, serving predictions via a Flask API, and providing a user-friendly interface using Streamlit.

## Features
- Fine-tune a BERT model for sentiment analysis.
- Serve predictions through a RESTful API using Flask.
- Provide a Streamlit-based web interface for user interaction.
- Preprocess and tokenize text data for model training and inference.
- Save and load trained models for reuse.

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Installation
1. Clone the repository:
   ```bash
    git remote add origin https://github.com/1003rashmitha/GenAI.git
    cd Sentiment_Analyzer
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure the `bert_trained` directory contains the necessary model files (e.g., `config.json`, `pytorch_model.bin`, etc.).

## Usage

### Training the Model
1. Open the `Bert_training.ipynb` notebook.
2. Follow the steps to preprocess data, fine-tune the BERT model, and save the trained model.

### Running the Flask API
1. Start the Flask server:
   ```bash
   python main.py
   ```
2. Access the API at `http://127.0.0.1:5000/`.

### Running the Streamlit App
1. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```
2. Access the Streamlit interface in your browser at `http://localhost:8501/`.

### Making Predictions via Streamlit
1. Enter your text in the input box provided in the Streamlit interface.
2. Click the "Predict" button to analyze the sentiment of the input text.

### Making Predictions via API
Send a POST request to the `/predict` endpoint with the following JSON payload:
```json
{
  "text": "Your input text here"
}
```
The API will return the sentiment of the input text.

## File Descriptions
- **Bert_training.ipynb**: Jupyter Notebook for training and fine-tuning the BERT model.
- **main.py**: Flask application for serving predictions.
- **app.py**: Streamlit application for user interaction.
- **requirements.txt**: List of Python dependencies.
- **reviews.xlsx**: Input dataset for training the model.
- **bert_trained/**: Directory containing the fine-tuned BERT model and tokenizer files.

## Dependencies
- Python 3.8+
- Transformers (Hugging Face)
- Datasets
- scikit-learn
- Flask
- Streamlit
- Torch

## License
This project is licensed under the MIT License.