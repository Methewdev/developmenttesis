import streamlit as st
import pandas as pd
import torch
import re
import plotly.express as px

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# ==================================================
# PAGE CONFIG
# ==================================================

st.set_page_config(
    page_title="Livin Sentiment & Emotion Analysis",
    page_icon="📊",
    layout="wide"
)

# ==================================================
# HUGGING FACE MODEL
# ==================================================

SENTIMENT_REPO = "envidevelopment/livin-sentiment"
EMOTION_REPO = "envidevelopment/livin-emotion"

# ==================================================
# LOAD MODEL
# ==================================================

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

    st.error(f"Gagal load model: {e}")
    st.stop()

# ==================================================
# LABEL
# ==================================================

sentiment_labels = sentiment_model.config.id2label
emotion_labels = emotion_model.config.id2label

# ==================================================
# CLEAN TEXT
# ==================================================

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

# ==================================================
# SENTIMENT
# ==================================================

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

# ==================================================
# EMOTION
# ==================================================

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

# ==================================================
# HEADER
# ==================================================

st.title("📊 Livin Sentiment & Emotion Analysis")

st.markdown(
"""
Analisis Sentimen dan Emosi menggunakan model IndoBERT
"""
)

# ==================================================
# INPUT
# ==================================================

text = st.text_area(
    "Masukkan Ulasan",
    height=180
)

# ==================================================
# BUTTON
# ==================================================

if st.button("Analisis"):

    if len(text.strip()) == 0:

        st.warning(
            "Silakan masukkan ulasan terlebih dahulu"
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
        # PREDICTION
        # ==========================================

        sent_pred, sent_probs = predict_sentiment(
            cleaned
        )

        emo_pred, emo_probs = predict_emotion(
            cleaned
        )

        # ==========================================
        # LABEL PREDICTION
        # ==========================================

        sent_label = sentiment_labels[sent_pred]

        emo_label = emotion_labels[emo_pred]

        # ==========================================
        # RESULT
        # ==========================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Sentimen")

            if sent_label == "Positive":

                st.success(sent_label)

            elif sent_label == "Neutral":

                st.info(sent_label)

            else:

                st.error(sent_label)

        with col2:

            st.subheader("Emosi")

            st.success(emo_label)

        # ==========================================
        # CLEANING RESULT
        # ==========================================

        st.subheader("Hasil Cleaning")

        st.code(cleaned)

        # ==========================================
        # TOKENIZATION
        # ==========================================

        st.subheader("Tokenisasi IndoBERT")

        st.write(tokens[:100])

        # ==========================================
        # SENTIMENT CHART
        # ==========================================

        sent_df = pd.DataFrame({

            "Label": [
                sentiment_labels[i]
                for i in range(
                    len(sent_probs)
                )
            ],

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

            "Label": [
                emotion_labels[i]
                for i in range(
                    len(emo_probs)
                )
            ],

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
            "Detail Probabilitas Sentimen"
        )

        st.dataframe(
            sent_df,
            use_container_width=True
        )

        st.subheader(
            "Detail Probabilitas Emosi"
        )

        st.dataframe(
            emo_df,
            use_container_width=True
        )
