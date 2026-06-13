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
def load_model():

    MODEL_NAME = "username/model-sentiment"

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME
    )

    return tokenizer, model


tokenizer, model = load_model()

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

        st.warning(
            "Masukkan ulasan terlebih dahulu"
        )

    else:

        # -----------------------------------------
        # PREPROCESSING
        # -----------------------------------------

        cleaned = clean_text(text)

        # -----------------------------------------
        # TOKENIZATION
        # -----------------------------------------

        tokens = tokenizer.tokenize(cleaned)

        token_ids = tokenizer.convert_tokens_to_ids(
            tokens
        )

        # -----------------------------------------
        # PREDICTION
        # -----------------------------------------

        pred, probs = predict_sentiment(
            cleaned
        )

        label = id2label[pred]

        # -----------------------------------------
        # RESULT
        # -----------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.subheader(
                "Preprocessing"
            )

            st.write(cleaned)

            st.subheader(
                "Tokenisasi"
            )

            st.write(tokens[:50])

        with col2:

            st.subheader(
                "Hasil Prediksi"
            )

            if label == "Positif":

                st.success(
                    f"Sentimen : {label}"
                )

            else:

                st.error(
                    f"Sentimen : {label}"
                )

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
