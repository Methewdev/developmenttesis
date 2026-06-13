# app.py

```python
import streamlit as st
import pandas as pd
import torch
import re
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Livin Sentiment & Emotion Dashboard",
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

    st.error(f"Gagal memuat model: {e}")
    st.stop()

# =====================================================
# LABEL
# =====================================================

sentiment_labels = sentiment_model.config.id2label
emotion_labels = emotion_model.config.id2label

# =====================================================
# PREPROCESSING
# =====================================================

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

# =====================================================
# PREDICT SENTIMENT
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
# PREDICT EMOTION
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
# DATASET PREDICTION
# =====================================================

def predict_dataset(text):

    text = clean_text(text)

    sent_pred, _ = predict_sentiment(text)
    emo_pred, _ = predict_emotion(text)

    return pd.Series({
        "sentiment": sentiment_labels[sent_pred],
        "emotion": emotion_labels[emo_pred]
    })

# =====================================================
# SIDEBAR
# =====================================================

menu = st.sidebar.radio(
    "Menu",
    [
        "🏠 Home",
        "✍️ Analisis Ulasan",
        "📁 Analisis Dataset"
    ]
)

# =====================================================
# HOME
# =====================================================

if menu == "🏠 Home":

    st.title("📊 Livin Sentiment & Emotion Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Jumlah Label Sentiment",
            sentiment_model.config.num_labels
        )

    with col2:
        st.metric(
            "Jumlah Label Emotion",
            emotion_model.config.num_labels
        )

    st.subheader("Sentiment Labels")
    st.json(sentiment_labels)

    st.subheader("Emotion Labels")
    st.json(emotion_labels)

# =====================================================
# SINGLE ANALYSIS
# =====================================================

elif menu == "✍️ Analisis Ulasan":

    st.title("✍️ Analisis Ulasan")

    text = st.text_area(
        "Masukkan Ulasan",
        height=180
    )

    if st.button("Analisis"):

        cleaned = clean_text(text)

        tokens = sentiment_tokenizer.tokenize(
            cleaned
        )

        token_ids = sentiment_tokenizer.convert_tokens_to_ids(
            tokens
        )

        sent_pred, sent_probs = predict_sentiment(
            cleaned
        )

        emo_pred, emo_probs = predict_emotion(
            cleaned
        )

        sent_label = sentiment_labels[sent_pred]
        emo_label = emotion_labels[emo_pred]

        sent_conf = max(sent_probs) * 100
        emo_conf = max(emo_probs) * 100

        col1, col2 = st.columns(2)

        with col1:
            st.success(
                f"Sentiment : {sent_label}"
            )

        with col2:
            st.info(
                f"Emotion : {emo_label}"
            )

        st.subheader("Preprocessing")

        st.write("Original Text")
        st.code(text)

        st.write("Cleaned Text")
        st.code(cleaned)

        st.subheader("Tokenization")

        st.write(tokens)

        st.subheader("Token IDs")

        st.write(token_ids)

        sent_df = pd.DataFrame({
            "Label":[
                sentiment_labels[i]
                for i in range(len(sent_probs))
            ],
            "Probability":sent_probs
        })

        emo_df = pd.DataFrame({
            "Label":[
                emotion_labels[i]
                for i in range(len(emo_probs))
            ],
            "Probability":emo_probs
        })

        st.subheader("Distribusi Sentiment")

        fig_sent = px.bar(
            sent_df,
            x="Label",
            y="Probability",
            text_auto=".2%"
        )

        st.plotly_chart(
            fig_sent,
            use_container_width=True
        )

        st.subheader("Distribusi Emotion")

        fig_emo = px.bar(
            emo_df,
            x="Label",
            y="Probability",
            text_auto=".2%"
        )

        st.plotly_chart(
            fig_emo,
            use_container_width=True
        )

        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sent_conf,
                title={
                    "text":"Sentiment Confidence"
                }
            )
        )

        st.plotly_chart(
            gauge,
            use_container_width=True
        )

# =====================================================
# DATASET ANALYSIS
# =====================================================

elif menu == "📁 Analisis Dataset":

    st.title("📁 Analisis Dataset")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file:

        df = pd.read_csv(
            uploaded_file
        )

        st.dataframe(
            df.head(),
            use_container_width=True
        )

        text_column = st.selectbox(
            "Pilih Kolom Ulasan",
            df.columns
        )

        if st.button(
            "🚀 Analisis Dataset"
        ):

            with st.spinner(
                "Sedang memproses..."
            ):

                result_df = df[
                    text_column
                ].apply(
                    predict_dataset
                )

                df = pd.concat(
                    [df, result_df],
                    axis=1
                )

            st.success(
                "Analisis selesai"
            )

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

            all_text = " ".join(
                df[text_column]
                .astype(str)
            )

            top_words = Counter(
                all_text.split()
            ).most_common(10)

            word_df = pd.DataFrame(
                top_words,
                columns=[
                    "Word",
                    "Frequency"
                ]
            )

            fig_word = px.bar(
                word_df,
                x="Word",
                y="Frequency",
                title="Top 10 Words"
            )

            st.plotly_chart(
                fig_word,
                use_container_width=True
            )

            st.dataframe(
                df,
                use_container_width=True
            )

            csv = df.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇ Download Hasil",
                csv,
                "hasil_analisis.csv",
                "text/csv"
            )
```
