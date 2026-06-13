import streamlit as st
import pandas as pd
import torch
import re
import plotly.express as px

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen Mobile Banking",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_models():

    sentiment_repo = "envidevelopment/livin-sentiment"
    emotion_repo = "envidevelopment/livin-emotion"

    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        sentiment_repo
    )

    sentiment_model = (
        AutoModelForSequenceClassification
        .from_pretrained(sentiment_repo)
    )

    emotion_tokenizer = AutoTokenizer.from_pretrained(
        emotion_repo
    )

    emotion_model = (
        AutoModelForSequenceClassification
        .from_pretrained(emotion_repo)
    )

    sentiment_model.eval()
    emotion_model.eval()

    return (
        sentiment_tokenizer,
        sentiment_model,
        emotion_tokenizer,
        emotion_model
    )

(
    sentiment_tokenizer,
    sentiment_model,
    emotion_tokenizer,
    emotion_model
) = load_models()

# =====================================================
# LABEL
# =====================================================

id2label = {
    0: "Negatif",
    1: "Positif"
}

# =====================================================
# PREPROCESSING
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"@\w+", "", text)

    text = re.sub(r"#[A-Za-z0-9_]+", "", text)

    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()

# =====================================================
# PREDICT
# =====================================================

def predict_sentiment(text):

    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        )

        pred = torch.argmax(
            probs,
            dim=1
        ).item()

    return pred, probs.squeeze().tolist()

# =====================================================
# HEADER
# =====================================================

st.title("📊 Analisis Sentimen Mobile Banking")

st.write(
    "Prediksi sentimen menggunakan model IndoBERT"
)

# =====================================================
# INPUT
# =====================================================

text = st.text_area(
    "Masukkan Ulasan",
    height=150
)

# =====================================================
# BUTTON
# =====================================================

if st.button("Analisis"):

    if text.strip() == "":
        st.warning("Masukkan ulasan terlebih dahulu")

    else:

        cleaned = clean_text(text)

        st.write("Hasil Cleaning:")
        st.write(cleaned)

        tokens = tokenizer.tokenize(cleaned)

        st.write("Tokens:")
        st.write(tokens[:50])
        # -----------------------------------------
        # PROBABILITY
        # -----------------------------------------

        prob_df = pd.DataFrame({
            "Kelas": [
                "Negatif",
                "Positif"
            ],
            "Probabilitas": probs
        })

        fig = px.bar(
            prob_df,
            x="Kelas",
            y="Probabilitas",
            text="Probabilitas"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        st.subheader(
            "Probabilitas Tiap Kelas"
        )

        st.dataframe(
            prob_df,
            use_container_width=True
        )
