import streamlit as st
import pandas as pd
import numpy as np
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Livin Review Analysis",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_models():

    sentiment_model_name = "envidevelopment/livin-sentiment"
    emotion_model_name = "envidevelopment/livin_emotion"

    sentiment_tokenizer = AutoTokenizer.from_pretrained(
        sentiment_model_name
    )

    sentiment_model = (
        AutoModelForSequenceClassification
        .from_pretrained(sentiment_model_name)
    )

    emotion_tokenizer = AutoTokenizer.from_pretrained(
        emotion_model_name
    )

    emotion_model = (
        AutoModelForSequenceClassification
        .from_pretrained(emotion_model_name)
    )

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

sentiment_labels = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

emotion_labels = {
    0: "Anger",
    1: "Fear",
    2: "Happy",
    3: "Love",
    4: "Sadness"
}

# =====================================================
# PREPROCESSING
# =====================================================

def clean_text(text):

    text = str(text)

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"www\S+", "", text)

    text = re.sub(r"@[A-Za-z0-9_]+", "", text)

    text = re.sub(r"#", "", text)

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
# PREDICTION
# =====================================================

def predict_sentiment(text):

    inputs = sentiment_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = sentiment_model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        ).numpy()[0]

    pred = np.argmax(probs)

    return (
        sentiment_labels[pred],
        probs
    )


def predict_emotion(text):

    inputs = emotion_tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = emotion_model(**inputs)

        probs = torch.softmax(
            outputs.logits,
            dim=1
        ).numpy()[0]

    pred = np.argmax(probs)

    return (
        emotion_labels[pred],
        probs
    )

# =====================================================
# TITLE
# =====================================================

st.title("📱 Livin Review Analysis")
st.markdown(
    """
Analisis Sentimen dan Emosi Ulasan Pengguna
menggunakan IndoBERT
"""
)

# =====================================================
# INPUT
# =====================================================

review = st.text_area(
    "Masukkan Ulasan",
    height=150,
    placeholder="Contoh: Aplikasi sangat membantu transaksi saya"
)

# =====================================================
# BUTTON
# =====================================================

if st.button("Analisis"):

    if review.strip() == "":

        st.warning(
            "Masukkan ulasan terlebih dahulu"
        )

    else:

        # ==================================
        # STEP 1
        # ==================================

        st.subheader("1️⃣ Teks Asli")

        st.info(review)

        # ==================================
        # STEP 2
        # ==================================

        cleaned = clean_text(review)

        st.subheader("2️⃣ Preprocessing")

        st.code(cleaned)

        # ==================================
        # STEP 3
        # ==================================

        tokens = sentiment_tokenizer.tokenize(
            cleaned
        )

        st.subheader("3️⃣ Tokenisasi")

        st.write(tokens)

        # ==================================
        # STEP 4
        # ==================================

        sentiment_result, sentiment_probs = (
            predict_sentiment(cleaned)
        )

        emotion_result, emotion_probs = (
            predict_emotion(cleaned)
        )

        # ==================================
        # STEP 5
        # ==================================

        st.subheader("4️⃣ Hasil Sentiment")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Sentiment",
                sentiment_result
            )

        with col2:

            st.metric(
                "Confidence",
                f"{max(sentiment_probs)*100:.2f}%"
            )

        sentiment_df = pd.DataFrame({
            "Sentiment":
            [
                "Negative",
                "Neutral",
                "Positive"
            ],
            "Probability":
            sentiment_probs
        })

        st.bar_chart(
            sentiment_df.set_index(
                "Sentiment"
            )
        )

        # ==================================
        # STEP 6
        # ==================================

        st.subheader("5️⃣ Hasil Emotion")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Emotion",
                emotion_result
            )

        with col2:

            st.metric(
                "Confidence",
                f"{max(emotion_probs)*100:.2f}%"
            )

        emotion_df = pd.DataFrame({
            "Emotion":
            [
                "Anger",
                "Fear",
                "Happy",
                "Love",
                "Sadness"
            ],
            "Probability":
            emotion_probs
        })

        st.bar_chart(
            emotion_df.set_index(
                "Emotion"
            )
        )

        # ==================================
        # STEP 7
        # ==================================

        st.subheader("6️⃣ Ringkasan")

        summary = pd.DataFrame({
            "Review":[review],
            "Sentiment":[sentiment_result],
            "Emotion":[emotion_result]
        })

        st.dataframe(
            summary,
            use_container_width=True
        )
