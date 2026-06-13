import streamlit as st
import pandas as pd
import numpy as np
import torch
import re

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ======================================================
# PAGE CONFIG
# ======================================================

st.set_page_config(
    page_title="Livin Sentiment & Emotion Dashboard",
    page_icon="📊",
    layout="wide"
)

# ======================================================
# MODEL REPOSITORY
# ======================================================

SENTIMENT_REPO = "envidevelopment/livin-sentiment"
EMOTION_REPO = "envidevelopment/livin-emotion"

# ======================================================
# LOAD MODEL
# ======================================================

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

    st.error(f"Gagal memuat model: {e}")
    st.stop()

# ======================================================
# LABEL
# ======================================================

sentiment_labels = sentiment_model.config.id2label
emotion_labels = emotion_model.config.id2label

# ======================================================
# CLEAN TEXT
# ======================================================

def clean_text(text):

    text = str(text).lower()

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

# ======================================================
# SENTIMENT
# ======================================================

def predict_sentiment(text):

    inputs = sentiment_tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
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

# ======================================================
# EMOTION
# ======================================================

def predict_emotion(text):

    inputs = emotion_tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
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

# ======================================================
# BATCH PREDICTION
# ======================================================

def predict_dataset(text):

    text = clean_text(text)

    sent_pred, _ = predict_sentiment(text)
    emo_pred, _ = predict_emotion(text)

    return (
        sentiment_labels[sent_pred],
        emotion_labels[emo_pred]
    )

# ======================================================
# HEADER
# ======================================================

st.title("📊 Livin Sentiment & Emotion Dashboard")

st.markdown(
"""
Analisis Sentimen dan Emosi Ulasan Mobile Banking
menggunakan IndoBERT.
"""
)

# ======================================================
# SINGLE PREDICTION
# ======================================================

st.header("✍️ Analisis Satu Ulasan")

text = st.text_area(
    "Masukkan Ulasan",
    height=150
)

if st.button("Analisis Ulasan"):

    if text.strip() == "":

        st.warning(
            "Masukkan ulasan terlebih dahulu."
        )

    else:

        cleaned = clean_text(text)

        sent_pred, sent_probs = predict_sentiment(
            cleaned
        )

        emo_pred, emo_probs = predict_emotion(
            cleaned
        )

        sent_label = sentiment_labels[sent_pred]
        emo_label = emotion_labels[emo_pred]

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Sentimen")

            st.success(sent_label)

        with col2:

            st.subheader("Emosi")

            st.info(emo_label)

        st.subheader("Hasil Cleaning")

        st.code(cleaned)

        st.subheader("Tokenisasi")

        st.write(
            sentiment_tokenizer.tokenize(cleaned)[:50]
        )

# ======================================================
# UPLOAD DATASET
# ======================================================

st.divider()

st.header("📁 Upload Dataset")

uploaded_file = st.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview Dataset")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

    if "content" not in df.columns:

        st.error(
            "Kolom 'content' tidak ditemukan."
        )

        st.stop()

    if st.button("🚀 Analisis Dataset"):

        with st.spinner(
            "Sedang menganalisis dataset..."
        ):

            results = df["content"].apply(
                predict_dataset
            )

            df["sentiment"] = results.apply(
                lambda x: x[0]
            )

            df["emotion"] = results.apply(
                lambda x: x[1]
            )

        st.success(
            "Analisis selesai."
        )

        # ==========================================
        # METRIC
        # ==========================================

        st.header("📈 Statistik")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Ulasan",
            len(df)
        )

        col2.metric(
            "Sentimen Dominan",
            df["sentiment"].mode()[0]
        )

        col3.metric(
            "Emosi Dominan",
            df["emotion"].mode()[0]
        )

        # ==========================================
        # SENTIMENT CHART
        # ==========================================

        st.header("📊 Distribusi Sentimen")

        sent_count = (
            df["sentiment"]
            .value_counts()
            .reset_index()
        )

        sent_count.columns = [
            "Sentiment",
            "Total"
        ]

        fig_sent = px.pie(
            sent_count,
            names="Sentiment",
            values="Total",
            hole=0.4
        )

        st.plotly_chart(
            fig_sent,
            use_container_width=True
        )

        # ==========================================
        # EMOTION CHART
        # ==========================================

        st.header("😊 Distribusi Emosi")

        emo_count = (
            df["emotion"]
            .value_counts()
            .reset_index()
        )

        emo_count.columns = [
            "Emotion",
            "Total"
        ]

        fig_emo = px.bar(
            emo_count,
            x="Emotion",
            y="Total",
            text_auto=True
        )

        st.plotly_chart(
            fig_emo,
            use_container_width=True
        )

        # ==========================================
        # WORD CLOUD
        # ==========================================

        st.header("☁️ Word Cloud")

        all_text = " ".join(
            df["content"]
            .astype(str)
            .tolist()
        )

        wc = WordCloud(
            width=1200,
            height=600,
            background_color="white"
        ).generate(all_text)

        fig, ax = plt.subplots(
            figsize=(12,6)
        )

        ax.imshow(wc)

        ax.axis("off")

        st.pyplot(fig)

        # ==========================================
        # TABLE
        # ==========================================

        st.header("📋 Hasil Prediksi")

        st.dataframe(
            df,
            use_container_width=True
        )

        # ==========================================
        # DOWNLOAD
        # ==========================================

        csv = df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="⬇️ Download Hasil CSV",
            data=csv,
            file_name="hasil_analisis_livin.csv",
            mime="text/csv"
        )
