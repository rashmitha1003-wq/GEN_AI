# Career Gap Analyzer

An AI-powered application that analyzes a candidate's resume against a job description, identifies skill gaps, and generates a personalized learning roadmap.

# 🚀 Live Demo
**[Try it here](https://career-grow-a078.onrender.com)**

> Hosted on a free tier, so the first load after inactivity can take up to a minute while the server wakes up. Please give it a moment.

## Overview

Career Gap Analyzer helps job seekers understand how well their current skills align with a target job description.

The application:

- Extracts text from PDF and DOCX resumes
- Analyzes resume skills against job requirements using an LLM
- Identifies matched, partially matched, and missing skills
- Calculates an overall skill match score
- Provides specific skill-level gap analysis
- Generates a personalized learning roadmap for missing skills
- Provides resume improvement suggestions

## Tech Stack

- Python
- Streamlit
- FastAPI
- LangChain
- LangChain Groq
- Pydantic
- Groq LLM
- PyPDF
- python-docx
- Docker

## Project Structure

- [main.py](main.py) — FastAPI backend for resume analysis
- [app.py](app.py) — Streamlit frontend for uploading resumes and viewing results
- [analyzer.py](analyzer.py) — LLM-based resume-job comparison logic
- [extractor.py](extractor.py) — Resume text extraction for PDF and DOCX files
- [schema.py](schema.py) — Pydantic data models for analysis and roadmap output
- [tutor.py](tutor.py) — Learning roadmap generation for missing skills
- [requirements.txt](requirements.txt) — Python dependencies
- [Dockerfile](Dockerfile) — Container setup for running both services

## Run Locally

1. Clone the repository:
```bash
git clone <repository-url>
cd career-gap-analyser
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Create a .env file
   ```bash
   GROQ_API_KEY = your_key
   ```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Start the backend:
```bash
uvicorn main:app --reload
```
The FastAPI backend will run at:

http://127.0.0.1:8000

5. Start the frontend in a second terminal:
```bash
streamlit run app.py
```

## Docker
The application is containerized to provide a consistent environment and a foundation for future deployment.
Build and run the project with Docker:

```bash
docker build -t career-gap-analyzer .
docker run -p 8000:8000 -p 8501:8501 --env-file .env career-gap-analyzer
```

The Streamlit application will be available at:
http://localhost:8501

The FastAPI backend will be available at:
http://localhost:8000
               
