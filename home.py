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

    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
        sentiment_model_name
    )

    emotion_tokenizer = AutoTokenizer.from_pretrained(
        emotion_model_name
    )

    emotion_model = AutoModelForSequenceClassification.from_pretrained(
        emotion_model_name
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
# SENTIMENT
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
        ).cpu().numpy()[0]

    pred = int(np.argmax(probs))

    try:
        label = sentiment_model.config.id2label[pred]
    except:
        label = sentiment_model.config.id2label[str(pred)]

    return label, probs

# =====================================================
# EMOTION
# =====================================================

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
        ).cpu().numpy()[0]

    pred = int(np.argmax(probs))

    try:
        label = emotion_model.config.id2label[pred]
    except:
        label = emotion_model.config.id2label[str(pred)]

    return label, probs

# =====================================================
# TITLE
# =====================================================

st.title("📱 Livin Review Analysis")

st.markdown("""
Analisis Sentimen dan Emosi Ulasan Pengguna
menggunakan IndoBERT + Hugging Face
""")

# =====================================================
# DEBUG MODEL
# =====================================================

with st.expander("🔍 Informasi Model"):

    st.write("Sentiment Labels:")
    st.json(sentiment_model.config.id2label)

    st.write("Emotion Labels:")
    st.json(emotion_model.config.id2label)

# =====================================================
# INPUT
# =====================================================

review = st.text_area(
    "Masukkan Ulasan",
    height=150,
    placeholder="Contoh: Aplikasi sangat membantu transaksi saya"
)

# =====================================================
# ANALISIS
# =====================================================

if st.button("🚀 Analisis"):

    if not review.strip():

        st.warning(
            "Masukkan ulasan terlebih dahulu."
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

        st.subheader("2️⃣ Hasil Preprocessing")

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

        sentiment_result, sentiment_probs = predict_sentiment(
            cleaned
        )

        emotion_result, emotion_probs = predict_emotion(
            cleaned
        )

        # ==================================
        # SENTIMENT
        # ==================================

        st.subheader("4️⃣ Hasil Sentimen")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Sentiment",
                sentiment_result
            )

        with col2:

            st.metric(
                "Confidence",
                f"{np.max(sentiment_probs)*100:.2f}%"
            )

        sentiment_labels = []

        try:
            for i in range(len(sentiment_probs)):
                sentiment_labels.append(
                    sentiment_model.config.id2label[i]
                )
        except:
            for i in range(len(sentiment_probs)):
                sentiment_labels.append(
                    sentiment_model.config.id2label[str(i)]
                )

        sentiment_df = pd.DataFrame({
            "Label": sentiment_labels,
            "Probability": sentiment_probs
        })

        st.bar_chart(
            sentiment_df.set_index(
                "Label"
            )
        )

        st.dataframe(
            sentiment_df,
            use_container_width=True
        )

        # ==================================
        # EMOTION
        # ==================================

        st.subheader("5️⃣ Hasil Emosi")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Emotion",
                emotion_result
            )

        with col2:

            st.metric(
                "Confidence",
                f"{np.max(emotion_probs)*100:.2f}%"
            )

        emotion_labels = []

        try:
            for i in range(len(emotion_probs)):
                emotion_labels.append(
                    emotion_model.config.id2label[i]
                )
        except:
            for i in range(len(emotion_probs)):
                emotion_labels.append(
                    emotion_model.config.id2label[str(i)]
                )

        emotion_df = pd.DataFrame({
            "Label": emotion_labels,
            "Probability": emotion_probs
        })

        st.bar_chart(
            emotion_df.set_index(
                "Label"
            )
        )

        st.dataframe(
            emotion_df,
            use_container_width=True
        )

        # ==================================
        # SUMMARY
        # ==================================

        st.subheader("6️⃣ Ringkasan")

        summary = pd.DataFrame({
            "Review": [review],
            "Sentiment": [sentiment_result],
            "Emotion": [emotion_result]
        })

        st.dataframe(
            summary,
            use_container_width=True
        )

        # ==================================
        # RAW PROBABILITY
        # ==================================

        with st.expander("📊 Detail Probabilitas"):

            st.write("Sentiment Probability")
            st.write(sentiment_probs)

            st.write("Emotion Probability")
            st.write(emotion_probs)
