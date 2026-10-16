#!/bin/sh
uvicorn main:app --host 127.0.0.1 --port 8000 &
streamlit run app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true