import streamlit as st
import pandas as pd
import torch.nn.functional as F
import torch
import plotly.express as px

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

# =====================================================
# CONFIG
# =====================================================

st.set_page_config(
    page_title="Emotion AI",
    page_icon="🧠",
    layout="wide"
)

# =====================================================
# STYLE
# =====================================================

st.markdown("""
<style>

[data-testid="stAppViewContainer"]{
    background-color:#020B1C;
}

[data-testid="stSidebar"]{
    background-color:#09152D;
}

h1,h2,h3,h4,h5,h6,p,label{
    color:white;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# DEVICE
# =====================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# =====================================================
# MODEL LOADER
# =====================================================

@st.cache_resource
def load_sentiment_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "envidevelopment/livin_sentiment"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "envidevelopment/livin_emotion"
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model


@st.cache_resource
def load_emotion_model():

    tokenizer = AutoTokenizer.from_pretrained(
        "envidevelopment/emotion_model"
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        "envidevelopment/emotion_model"
    )

    model.to(DEVICE)
    model.eval()

    return tokenizer, model

# =====================================================
# PREDICTION
# =====================================================

def predict_sentiment(text):

    tokenizer, model = load_sentiment_model()

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    confidence = torch.max(probs).item() * 100

    pred = torch.argmax(probs, dim=1).item()

    label_map = {
        0: "Negatif",
        1: "Netral",
        2: "Positif"
    }

    return (
        label_map.get(pred, "Unknown"),
        round(confidence, 2)
    )
def predict_emotion(text):

    tokenizer, model = load_emotion_model()

    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)

    confidence = torch.max(probs).item() * 100

    pred = torch.argmax(probs, dim=1).item()

    emotion_map = {
        0: "😡 Anger",
        1: "😨 Fear",
        2: "😊 Happy",
        3: "❤️ Love",
        4: "😢 Sadness"
    }

    return (
        emotion_map.get(pred, "❓ Unknown"),
        round(confidence, 2)
    )
# =====================================================
# SESSION
# =====================================================

if "single_result" not in st.session_state:
    st.session_state.single_result = None

if "input_text" not in st.session_state:
    st.session_state.input_text = ""

if "result_df" not in st.session_state:
    st.session_state.result_df = None
# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("🧠 Emotion AI")

menu = st.sidebar.radio(
    "Menu",
    [
        "Dashboard",
        "Analisis Satuan",
        "Bulk CSV"
    ]
)

# =====================================================
# DASHBOARD
# =====================================================

if menu == "Dashboard":

    st.title("📊 Dashboard Analisis")

    top_col1, top_col2 = st.columns([8, 2])

    with top_col2:

        if st.button(
            "🔄 Refresh",
            use_container_width=True
        ):

            st.session_state.result_df = None

            st.rerun()

    if st.session_state.result_df is None:

        st.info(
            "Silakan proses data pada menu Bulk CSV terlebih dahulu."
        )

    else:

        df = st.session_state.result_df

        total = len(df)

        positif = len(
            df[df["Sentiment"] == "Positif"]
        )

        negatif = len(
            df[df["Sentiment"] == "Negatif"]
        )

        netral = len(
            df[df["Sentiment"] == "Netral"]
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Total Data",
            total
        )

        c2.metric(
            "Positif",
            positif
        )

        c3.metric(
            "Negatif",
            negatif
        )

        c4.metric(
            "Netral",
            netral
        )

        st.markdown("---")

        chart1, chart2 = st.columns(2)

        with chart1:

            sentiment_count = (
                df["Sentiment"]
                .value_counts()
                .reset_index()
            )

            sentiment_count.columns = [
                "Sentiment",
                "Jumlah"
            ]

            fig_sentiment = px.pie(
                sentiment_count,
                names="Sentiment",
                values="Jumlah",
                title="Distribusi Sentimen"
            )

            st.plotly_chart(
                fig_sentiment,
                use_container_width=True
            )

        with chart2:

            emotion_count = (
                df["Emotion"]
                .value_counts()
                .reset_index()
            )

            emotion_count.columns = [
                "Emotion",
                "Jumlah"
            ]

            fig_emotion = px.bar(
                emotion_count,
                x="Emotion",
                y="Jumlah",
                title="Distribusi Emosi"
            )

            st.plotly_chart(
                fig_emotion,
                use_container_width=True
            )

        st.markdown("---")

        st.subheader(
            "📋 Hasil Analisis"
        )

        st.dataframe(
            df,
            use_container_width=True
        )

# =====================================================
# ANALISIS SATUAN
# =====================================================
elif menu == "Analisis Satuan":

    top1, top2 = st.columns([8, 2])

    with top1:
        st.title("🔍 Analisis Sentimen & Emosi")

    with top2:

        if st.button(
            "🔄 Refresh",
            use_container_width=True
        ):

            st.session_state.single_result = None
            st.session_state.input_text = ""

            st.rerun()

    text = st.text_area(
        "Masukkan Ulasan",
        height=180,
        key="input_text"
    )

    if st.button("🚀 Analisis"):

        if not text.strip():

            st.warning(
                "Masukkan teks terlebih dahulu"
            )

        else:

            try:

                with st.spinner(
                    "🧠 Sedang menganalisis..."
                ):

                    sentiment, sentiment_score = predict_sentiment(
                        text
                    )

                    emotion, emotion_score = predict_emotion(
                        text
                    )

                st.session_state.single_result = {
                    "sentiment": sentiment,
                    "sentiment_score": sentiment_score,
                    "emotion": emotion,
                    "emotion_score": emotion_score
                }

            except Exception as e:

                st.error(
                    f"Error : {e}"
                )

    if st.session_state.single_result is not None:

        result = st.session_state.single_result

        st.markdown("## 📋 Hasil Analisis")

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Sentimen",
                result["sentiment"]
            )

        with c2:
            st.metric(
                "Confidence",
                f'{result["sentiment_score"]:.2f}%'
            )

        with c3:
            st.metric(
                "Emosi",
                result["emotion"]
            )

        with c4:
            st.metric(
                "Confidence",
                f'{result["emotion_score"]:.2f}%'
            )

# =====================================================
# BULK CSV
# =====================================================

elif menu == "Bulk CSV":

    st.title("📂 Bulk CSV")

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        try:

            df = pd.read_csv(
                uploaded_file,
                encoding="latin1",
                sep=None,
                engine="python",
                on_bad_lines="skip"
            )

            st.dataframe(
                df.head()
            )

            text_col = st.selectbox(
                "Pilih Kolom Teks",
                df.columns
            )

            if st.button("Proses Data"):

                sentiments = []
                emotions = []

                progress = st.progress(0)

                for i, text in enumerate(df[text_col]):

                    sentiment_label, _ = predict_sentiment(
                        str(text)
                    )

                    emotion_label, _ = predict_emotion(
                        str(text)
                    )

                    # Hilangkan emoji pada hasil CSV
                    emotion_label = (
                        emotion_label
                        .replace("😡 ", "")
                        .replace("😨 ", "")
                        .replace("😊 ", "")
                        .replace("❤️ ", "")
                        .replace("😢 ", "")
                        .replace("❓ ", "")
                    )

                    sentiments.append(
                        sentiment_label
                    )

                    emotions.append(
                        emotion_label
                    )

                    progress.progress(
                        (i + 1) / len(df)
                    )

                df["Sentiment"] = sentiments
                df["Emotion"] = emotions

                st.session_state.result_df = df

                st.success(
                    "✅ Analisis selesai"
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
                    file_name="hasil_analisis.csv",
                    mime="text/csv"
                )

        except Exception as e:

            st.error(
                f"Error: {e}"
            )
