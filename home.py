
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
    page_icon="📱",
    layout="wide"
)

# =====================================================
# LOAD MODEL
# =====================================================

def load_models():

    sentiment_repo = "envidevelopment/livin-sentiment"
    emotion_repo = "envidevelopment/livin_emotion"

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
# PREPROCESSING
# =====================================================

def clean_text(text):

    text = str(text)

    text = re.sub(
        r"http\S+",
        "",
        text
    )

    text = re.sub(
        r"www\S+",
        "",
        text
    )

    text = re.sub(
        r"@[A-Za-z0-9_]+",
        "",
        text
    )

    text = re.sub(
        r"#",
        "",
        text
    )

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

        outputs = sentiment_model(
            **inputs
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1
        ).cpu().numpy()[0]

    pred = int(
        np.argmax(probs)
    )

    label = (
        sentiment_model
        .config
        .id2label
        .get(
            pred,
            sentiment_model
            .config
            .id2label
            .get(str(pred))
        )
    )

    return (
        label,
        probs,
        pred
    )

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

        outputs = emotion_model(
            **inputs
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1
        ).cpu().numpy()[0]

    pred = int(
        np.argmax(probs)
    )

    label = (
        emotion_model
        .config
        .id2label
        .get(
            pred,
            emotion_model
            .config
            .id2label
            .get(str(pred))
        )
    )

    return (
        label,
        probs,
        pred
    )

# =====================================================
# HEADER
# =====================================================

st.title(
    "📱 Livin Review Analysis"
)

st.markdown(
"""
Analisis Sentimen dan Emosi
Menggunakan IndoBERT Transformer
"""
)

# =====================================================
# MODEL INFO
# =====================================================

with st.expander(
    "🔍 Informasi Model"
):

    st.write(
        "Sentiment Repository"
    )

    st.write(
        sentiment_model.name_or_path
    )

    st.write(
        "Sentiment ID2LABEL"
    )

    st.json(
        sentiment_model
        .config
        .id2label
    )

    st.write(
        "Sentiment LABEL2ID"
    )

    st.json(
        sentiment_model
        .config
        .label2id
    )

    st.write(
        "Emotion Repository"
    )

    st.write(
        emotion_model.name_or_path
    )

    st.write(
        "Emotion ID2LABEL"
    )

    st.json(
        emotion_model
        .config
        .id2label
    )

    st.write(
        "Emotion LABEL2ID"
    )

    st.json(
        emotion_model
        .config
        .label2id
    )

# =====================================================
# INPUT
# =====================================================

review = st.text_area(
    "Masukkan Ulasan",
    height=200,
    placeholder="Contoh: aplikasi sangat membantu transaksi saya"
)

# =====================================================
# ANALISIS
# =====================================================

if st.button("🚀 Analisis"):

    if not review.strip():

        st.warning(
            "Masukkan ulasan terlebih dahulu"
        )

    else:

        cleaned = clean_text(
            review
        )

        st.subheader(
            "1️⃣ Teks Asli"
        )

        st.info(
            review
        )

        st.subheader(
            "2️⃣ Hasil Preprocessing"
        )

        st.code(
            cleaned
        )

        tokens = (
            sentiment_tokenizer
            .tokenize(cleaned)
        )

        st.subheader(
            "3️⃣ Tokenisasi"
        )

        st.write(
            tokens
        )

        (
            sentiment_result,
            sentiment_probs,
            sentiment_pred
        ) = predict_sentiment(
            cleaned
        )

        (
            emotion_result,
            emotion_probs,
            emotion_pred
        ) = predict_emotion(
            cleaned
        )

        # =====================================
        # DEBUG
        # =====================================

        st.subheader(
            "🔬 Debug Prediction"
        )

        st.write(
            "Sentiment Probabilities"
        )

        st.write(
            sentiment_probs
        )

        st.write(
            "Sentiment Argmax"
        )

        st.write(
            sentiment_pred
        )

        st.write(
            "Emotion Probabilities"
        )

        st.write(
            emotion_probs
        )

        st.write(
            "Emotion Argmax"
        )

        st.write(
            emotion_pred
        )

        # =====================================
        # SENTIMENT
        # =====================================

        st.subheader(
            "4️⃣ Hasil Sentimen"
        )

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

        sentiment_labels = [

            sentiment_model
            .config
            .id2label
            .get(
                i,
                sentiment_model
                .config
                .id2label
                .get(str(i))
            )

            for i in range(
                len(sentiment_probs)
            )
        ]

        sentiment_df = pd.DataFrame({

            "Label":
            sentiment_labels,

            "Probability":
            sentiment_probs

        })

        st.dataframe(
            sentiment_df,
            use_container_width=True
        )

        st.bar_chart(
            sentiment_df.set_index(
                "Label"
            )
        )

        # =====================================
        # EMOTION
        # =====================================

        st.subheader(
            "5️⃣ Hasil Emosi"
        )

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

        emotion_labels = [

            emotion_model
            .config
            .id2label
            .get(
                i,
                emotion_model
                .config
                .id2label
                .get(str(i))
            )

            for i in range(
                len(emotion_probs)
            )
        ]

        emotion_df = pd.DataFrame({

            "Label":
            emotion_labels,

            "Probability":
            emotion_probs

        })

        st.dataframe(
            emotion_df,
            use_container_width=True
        )

        st.bar_chart(
            emotion_df.set_index(
                "Label"
            )
        )

        # =====================================
        # SUMMARY
        # =====================================

        st.subheader(
            "6️⃣ Ringkasan"
        )

        summary = pd.DataFrame({

            "Review":
            [review],

            "Sentiment":
            [sentiment_result],

            "Emotion":
            [emotion_result]

        })

        st.dataframe(
            summary,
            use_container_width=True
        )
