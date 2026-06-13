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
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Analisis Sentimen & Emosi Livin",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# MODEL REPOSITORY
# =====================================================

SENTIMENT_REPO = "envidevelopment/livin-sentiment"
EMOTION_REPO = "envidevelopment/livin-emotion"

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_models():

    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        SENTIMENT_REPO
    )

    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        SENTIMENT_REPO
    )

    emotion_tokenizer = AutoTokenizer.from_pretrained(
        EMOTION_REPO
    )

    emotion_model = AutoModelForSequenceClassification.from_pretrained(
        EMOTION_REPO
    )

    return (
        sentiment_tokenizer,
        sentiment_model,
        emotion_tokenizer,
        emotion_model
    )


try:

    (
        sentiment_tokenizer,
        sentiment_model,
        emotion_tokenizer,
        emotion_model
    ) = load_models()

except Exception as e:

    st.error(f"Gagal load model : {e}")
    st.stop()

# =====================================================
# LABEL
# =====================================================

sentiment_labels = {
    0: "Negatif",
    1: "Positif"
}

emotion_labels = {
    0: "Marah",
    1: "Senang",
    2: "Sedih",
    3: "Frustrasi"
}

# =====================================================
# CLEANING
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()

# =====================================================
# SENTIMENT PREDICTION
# =====================================================

def predict_sentiment(text):

    inputs = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = sentiment_model(**inputs)

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
# EMOTION PREDICTION
# =====================================================

def predict_emotion(text):

    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():

        outputs = emotion_model(**inputs)

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

st.title("📊 Analisis Sentimen dan Emosi Livin")

st.markdown(
"""
Model:
- Sentiment Analysis
- Emotion Analysis

Berbasis IndoBERT
"""
)

# =====================================================
# INPUT
# =====================================================

text = st.text_area(
    "Masukkan Ulasan",
    height=200
)

# =====================================================
# BUTTON
# =====================================================

if st.button("Analisis"):

    if text.strip() == "":

        st.warning(
            "Masukkan ulasan terlebih dahulu"
        )

    else:

        # ==========================================
        # CLEANING
        # ==========================================

        cleaned = clean_text(text)

        # ==========================================
        # TOKENIZATION
        # ==========================================

        tokens = sentiment_tokenizer.tokenize(
            cleaned
        )

        # ==========================================
        # SENTIMENT
        # ==========================================

        sent_pred, sent_probs = predict_sentiment(
            cleaned
        )

        # ==========================================
        # EMOTION
        # ==========================================

        emo_pred, emo_probs = predict_emotion(
            cleaned
        )

        # ==========================================
        # RESULT
        # ==========================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Sentimen")

            st.success(
                sentiment_labels[sent_pred]
            )

        with col2:

            st.subheader("Emosi")

            st.info(
                emotion_labels[emo_pred]
            )

        # ==========================================
        # CLEANED TEXT
        # ==========================================

        st.subheader("Hasil Cleaning")

        st.write(cleaned)

        # ==========================================
        # TOKENS
        # ==========================================

        st.subheader("Tokenisasi IndoBERT")

        st.write(tokens[:100])

        # ==========================================
        # SENTIMENT CHART
        # ==========================================

        sent_df = pd.DataFrame({

            "Label": list(
                sentiment_labels.values()
            ),

            "Probabilitas": sent_probs
        })

        fig1 = px.bar(
            sent_df,
            x="Label",
            y="Probabilitas",
            title="Probabilitas Sentimen"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        # ==========================================
        # EMOTION CHART
        # ==========================================

        emo_df = pd.DataFrame({

            "Label": list(
                emotion_labels.values()
            ),

            "Probabilitas": emo_probs
        })

        fig2 = px.bar(
            emo_df,
            x="Label",
            y="Probabilitas",
            title="Probabilitas Emosi"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # ==========================================
        # TABLE
        # ==========================================

        st.subheader(
            "Detail Probabilitas Emosi"
        )

        st.dataframe(
            emo_df,
            use_container_width=True
        )
