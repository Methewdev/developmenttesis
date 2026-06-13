# app.py
# Livin Sentiment & Emotion Dashboard
# Generated for Streamlit deployment

import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import plotly.express as px
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Livin Review Analysis", page_icon="📱", layout="wide")

SENTIMENT_REPO = "envidevelopment/livin-sentiment"
EMOTION_REPO = "envidevelopment/livin-emotion"

@st.cache_resource
def load_models():
    stokenizer = AutoTokenizer.from_pretrained(SENTIMENT_REPO)
    smodel = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_REPO)
    etokenizer = AutoTokenizer.from_pretrained(EMOTION_REPO)
    emodel = AutoModelForSequenceClassification.from_pretrained(EMOTION_REPO)
    return stokenizer, smodel, etokenizer, emodel

sentiment_tokenizer, sentiment_model, emotion_tokenizer, emotion_model = load_models()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|@\w+|#', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def predict_sentiment(text):
    inputs = sentiment_tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    with torch.no_grad():
        probs = torch.softmax(sentiment_model(**inputs).logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return sentiment_model.config.id2label[pred], probs

def predict_emotion(text):
    inputs = emotion_tokenizer(text, return_tensors='pt', truncation=True, max_length=256)
    with torch.no_grad():
        probs = torch.softmax(emotion_model(**inputs).logits, dim=1).cpu().numpy()[0]
    pred = int(np.argmax(probs))
    return emotion_model.config.id2label[pred], probs

menu = st.sidebar.radio('Menu',['🏠 Dashboard','✍️ Single Analysis','📁 Dataset Analysis'])
st.title('📱 Livin Review Analysis')
