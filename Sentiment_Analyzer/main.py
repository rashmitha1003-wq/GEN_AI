from transformers import BertTokenizer, BertForSequenceClassification
import torch
from flask import Flask, jsonify, request
# import streamlit as st
# import requests

app=Flask(__name__)
model= BertForSequenceClassification.from_pretrained("./bert_trained")
tokenizer = BertTokenizer.from_pretrained("./bert_trained")

model.eval()

labels = {0: "negative", 1: "neutral", 2: "positive"}

@app.route('/')
def index():
    return "Sentiment Analysis app is running"

@app.route('/predict', methods=["POST"])
def predict():
    data = request.json
    text = data.get("text")
    if not text:
        return jsonify({"error":"Please provide the input"}) ,400

# Tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

# Get prediction
    with torch.no_grad():    #no_grad--> not to calculate gradient (declared when you are using the model for inference rather than training)
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = torch.argmax(logits, dim=1).item()
    
    return jsonify({
        "input":text,
        "sentiment":labels.get(predicted_class_id),
        })

if __name__=='__main__':
    app.run(debug=True)


