
import streamlit as st
import pandas as pd
import numpy as np
import re
import torch
import plotly.express as px

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="Livin Review Analysis",
    page_icon="📱",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

.metric-card {
    background-color:#f5f5f5;
    padding:20px;
    border-radius:15px;
    text-align:center;
}

.block-container {
    padding-top:2rem;
}

</style>
""", unsafe_allow_html=True)

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
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("📱 Livin Analysis")

    st.markdown("---")

    st.success("Sentiment Model Loaded")
    st.success("Emotion Model Loaded")

    st.markdown("---")

    st.subheader("Model Sentiment")
    st.caption(sentiment_model.name_or_path)

    st.subheader("Model Emotion")
    st.caption(emotion_model.name_or_path)

    st.markdown("---")

    st.info(
        """
        Analisis Sentimen dan Emosi
        menggunakan Transformer IndoBERT
        """
    )

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

        outputs = sentiment_model(
            **inputs
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1
        ).cpu().numpy()[0]

    pred = int(np.argmax(probs))

    label = (
        sentiment_model
        .config
        .id2label
        .get(pred)
    )

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

        outputs = emotion_model(
            **inputs
        )

        probs = torch.softmax(
            outputs.logits,
            dim=1
        ).cpu().numpy()[0]

    pred = int(np.argmax(probs))

    label = (
        emotion_model
        .config
        .id2label
        .get(pred)
    )

    return label, probs

# =====================================================
# HEADER
# =====================================================

st.markdown("""
<h1 style='text-align:center'>
📱 Livin Review Analysis
</h1>

<h4 style='text-align:center;color:gray'>
Sentiment & Emotion Analysis using IndoBERT
</h4>

<hr>
""", unsafe_allow_html=True)

# =====================================================
# INPUT
# =====================================================

review = st.text_area(
    "Masukkan Ulasan",
    height=200,
    placeholder="Contoh : Aplikasi sangat membantu transaksi saya..."
)

# =====================================================
# BUTTON
# =====================================================

if st.button("🚀 Analisis", use_container_width=True):

    if not review.strip():

        st.warning(
            "Masukkan ulasan terlebih dahulu"
        )

    else:

        cleaned = clean_text(review)

        tokens = (
            sentiment_tokenizer
            .tokenize(cleaned)
        )

        sentiment_result, sentiment_probs = (
            predict_sentiment(cleaned)
        )

        emotion_result, emotion_probs = (
            predict_emotion(cleaned)
        )

        sentiment_conf = (
            np.max(sentiment_probs) * 100
        )

        emotion_conf = (
            np.max(emotion_probs) * 100
        )

        # =====================================
        # RESULT
        # =====================================

        st.subheader("📊 Hasil Analisis")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Sentiment",
                sentiment_result
            )

            st.progress(
                int(sentiment_conf)
            )

            st.write(
                f"Confidence : {sentiment_conf:.2f}%"
            )

        with col2:

            st.metric(
                "Emotion",
                emotion_result
            )

            st.progress(
                int(emotion_conf)
            )

            st.write(
                f"Confidence : {emotion_conf:.2f}%"
            )

        # =====================================
        # NLP STATS
        # =====================================

        st.subheader("🔤 NLP Statistics")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Characters",
            len(review)
        )

        c2.metric(
            "Words",
            len(review.split())
        )

        c3.metric(
            "Tokens",
            len(tokens)
        )

        # =====================================
        # REVIEW
        # =====================================

        st.subheader("📝 Review")

        st.info(review)

        # =====================================
        # SENTIMENT CHART
        # =====================================

        sentiment_labels = [

            sentiment_model
            .config
            .id2label[i]

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

        st.subheader(
            "📈 Sentiment Distribution"
        )

        fig = px.bar(
            sentiment_df,
            x="Label",
            y="Probability",
            text="Probability"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =====================================
        # EMOTION CHART
        # =====================================

        emotion_labels = [

            emotion_model
            .config
            .id2label[i]

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

        st.subheader(
            "🎭 Emotion Distribution"
        )

        fig2 = px.pie(
            emotion_df,
            values="Probability",
            names="Label",
            hole=0.5
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # =====================================
        # TABLE RESULT
        # =====================================

        st.subheader(
            "📋 Ringkasan"
        )

        summary = pd.DataFrame({

            "Review":[review],

            "Sentiment":[
                sentiment_result
            ],

            "Emotion":[
                emotion_result
            ]

        })

        st.dataframe(
            summary,
            use_container_width=True
        )

        # =====================================
        # TECHNICAL DETAIL
        # =====================================

        with st.expander(
            "⚙️ Technical Details"
        ):

            st.write(
                "Clean Text"
            )

            st.code(cleaned)

            st.write(
                "Tokens"
            )

            st.write(tokens)

            st.write(
                "Sentiment Probability"
            )

            st.write(sentiment_df)

            st.write(
                "Emotion Probability"
            )

            st.write(emotion_df)
