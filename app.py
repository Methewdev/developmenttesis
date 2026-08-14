import streamlit as st
import pandas as pd
import numpy as np

from modules.scraper import scrape_reviews
from modules.preprocessing import preprocess_dataframe, preprocess_text
from modules.emotion import (
    load_emotion_model,
    predict_emotion,
    predict_batch
)
from modules.clustering import (
    perform_clustering,
    get_cluster_profile
)
from modules.retention import (
    get_retention_strategy,
    get_single_retention
)

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Livin' Emotion Analysis",
    page_icon="📊",
    layout="wide"
)

# =========================================================
# SESSION STATE
# =========================================================

if "reviews" not in st.session_state:
    st.session_state.reviews = None

if "processed_reviews" not in st.session_state:
    st.session_state.processed_reviews = None

if "predictions" not in st.session_state:
    st.session_state.predictions = None

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None

if "clustered_data" not in st.session_state:
    st.session_state.clustered_data = None

if "cluster_profile" not in st.session_state:
    st.session_state.cluster_profile = None

# =========================================================
# TITLE
# =========================================================

st.title("Livin' by Mandiri - Emotion Analysis")

st.caption(
    "Analisis Emosi Nasabah pada Ulasan Aplikasi Livin' by Mandiri "
    "Menggunakan Pendekatan Transformer sebagai Dasar Segmentasi "
    "dan Customer Retention"
)

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.header("Menu")

    menu = st.radio(
        "Pilih halaman",
        [
            "Dashboard",
            "Live Scraper",
            "Data Understanding",
            "Preprocessing",
            "Emotion Prediction",
            "Emotion Probability",
            "Customer Segmentation",
            "Customer Retention",
            "Single Analysis"
        ]
    )

    st.divider()

    st.caption("Model")
    st.write("IndoBERT")

    st.caption("Objek")
    st.write("Livin' by Mandiri")

# =========================================================
# LOAD MODEL
# =========================================================

@st.cache_resource
def get_model():

    tokenizer, model = load_emotion_model()

    return tokenizer, model


# =========================================================
# 1. LIVE SCRAPER
# =========================================================

if menu == "Live Scraper":

    st.header("1. Live Scraper")

    st.write(
        "Mengambil ulasan aplikasi secara langsung dari Google Play Store."
    )

    app_id = st.text_input(
        "Application ID",
        value="id.bmri.livin"
    )

    col1, col2 = st.columns(2)

    with col1:

        jumlah = st.number_input(
            "Jumlah review",
            min_value=10,
            max_value=5000,
            value=100,
            step=10
        )

    with col2:

        rating_filter = st.selectbox(
            "Filter rating",
            [
                "Semua",
                "1-2",
                "3",
                "4-5"
            ]
        )

    if st.button(
        "Mulai Scraping",
        type="primary",
        use_container_width=True
    ):

        with st.spinner("Mengambil review..."):

            try:

                df = scrape_reviews(
                    app_id=app_id,
                    count=jumlah
                )

                if rating_filter == "1-2":
                    df = df[df["rating"].between(1, 2)]

                elif rating_filter == "3":
                    df = df[df["rating"] == 3]

                elif rating_filter == "4-5":
                    df = df[df["rating"].between(4, 5)]

                st.session_state.reviews = df

                st.success(
                    f"Berhasil mengambil {len(df)} review."
                )

            except Exception as e:

                st.error(
                    f"Scraping gagal: {str(e)}"
                )

    if st.session_state.reviews is not None:

        df = st.session_state.reviews

        st.subheader("Hasil Scraping")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Jumlah Review",
            len(df)
        )

        col2.metric(
            "Rata-rata Rating",
            round(df["rating"].mean(), 2)
        )

        col3.metric(
            "Rating Terendah",
            df["rating"].min()
        )

        st.dataframe(
            df,
            use_container_width=True,
            height=500
        )

# =========================================================
# 2. DATA UNDERSTANDING
# =========================================================

elif menu == "Data Understanding":

    st.header("2. Data Understanding")

    df = st.session_state.reviews

    if df is None:

        st.warning(
            "Belum ada dataset. Jalankan Live Scraper terlebih dahulu."
        )

    else:

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Jumlah Data",
            len(df)
        )

        col2.metric(
            "Rata-rata Rating",
            round(df["rating"].mean(), 2)
        )

        col3.metric(
            "Rating Tertinggi",
            df["rating"].max()
        )

        col4.metric(
            "Rating Terendah",
            df["rating"].min()
        )

        st.divider()

        st.subheader("Distribusi Rating")

        rating_distribution = (
            df["rating"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            rating_distribution
        )

        st.subheader("Karakteristik Dataset")

        col1, col2 = st.columns(2)

        with col1:

            st.write("Jumlah karakter")

            df_character = df.copy()

            df_character["text_length"] = (
                df_character["content"]
                .astype(str)
                .str.len()
            )

            st.metric(
                "Rata-rata karakter",
                round(
                    df_character["text_length"].mean(),
                    2
                )
            )

        with col2:

            st.write("Jumlah kata")

            df_words = df.copy()

            df_words["word_count"] = (
                df_words["content"]
                .astype(str)
                .str.split()
                .str.len()
            )

            st.metric(
                "Rata-rata kata",
                round(
                    df_words["word_count"].mean(),
                    2
                )
            )

        st.subheader("Preview Dataset")

        st.dataframe(
            df.head(100),
            use_container_width=True
        )

# =========================================================
# 3. PREPROCESSING
# =========================================================

elif menu == "Preprocessing":

    st.header("3. Preprocessing")

    df = st.session_state.reviews

    if df is None:

        st.warning(
            "Belum ada dataset."
        )

    else:

        if st.button(
            "Jalankan Preprocessing",
            type="primary"
        ):

            with st.spinner(
                "Melakukan preprocessing..."
            ):

                processed = preprocess_dataframe(
                    df
                )

                st.session_state.processed_reviews = processed

            st.success(
                "Preprocessing selesai."
            )

        if st.session_state.processed_reviews is not None:

            processed = st.session_state.processed_reviews

            st.subheader(
                "Hasil Preprocessing"
            )

            st.dataframe(
                processed,
                use_container_width=True,
                height=500
            )

            st.subheader(
                "Tahapan Preprocessing"
            )

            stages = [
                "Cleaning",
                "Case Folding",
                "Normalization",
                "Repeated Character Handling",
                "Tokenization"
            ]

            for i, stage in enumerate(stages, 1):

                st.write(
                    f"**{i}. {stage}**"
                )

# =========================================================
# 4. EMOTION PREDICTION
# =========================================================

elif menu == "Emotion Prediction":

    st.header("4. Emotion Prediction")

    df = st.session_state.processed_reviews

    if df is None:

        st.warning(
            "Jalankan preprocessing terlebih dahulu."
        )

    else:

        if st.button(
            "Prediksi Emosi",
            type="primary"
        ):

            with st.spinner(
                "Melakukan emotion prediction..."
            ):

                tokenizer, model = get_model()

                result = predict_batch(
                    df,
                    tokenizer,
                    model
                )

                st.session_state.predictions = result

            st.success(
                "Prediksi emosi selesai."
            )

        if st.session_state.predictions is not None:

            result = st.session_state.predictions

            st.subheader(
                "Hasil Prediksi"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Jumlah Data",
                len(result)
            )

            col2.metric(
                "Confidence Rata-rata",
                f"{result['confidence'].mean():.2%}"
            )

            col3.metric(
                "Emosi Dominan",
                result["emotion"].value_counts().idxmax()
            )

            st.subheader(
                "Distribusi Emosi"
            )

            st.bar_chart(
                result["emotion"].value_counts()
            )

            st.subheader(
                "Detail Prediksi"
            )

            st.dataframe(
                result,
                use_container_width=True,
                height=500
            )

# =========================================================
# 5. EMOTION PROBABILITY
# =========================================================

elif menu == "Emotion Probability":

    st.header("5. Emotion Probability")

    result = st.session_state.predictions

    if result is None:

        st.warning(
            "Jalankan Emotion Prediction terlebih dahulu."
        )

    else:

        probability_columns = [
            "prob_senang",
            "prob_sedih",
            "prob_frustrasi",
            "prob_netral"
        ]

        available_columns = [
            col
            for col in probability_columns
            if col in result.columns
        ]

        st.subheader(
            "Rata-rata Emotion Probability"
        )

        averages = result[
            available_columns
        ].mean()

        cols = st.columns(
            len(available_columns)
        )

        for col, name in zip(
            cols,
            available_columns
        ):

            label = (
                name
                .replace("prob_", "")
                .capitalize()
            )

            col.metric(
                label,
                f"{averages[name]:.2%}"
            )

        st.divider()

        st.subheader(
            "Distribusi Probability"
        )

        st.dataframe(
            result[
                available_columns
            ].describe().T,
            use_container_width=True
        )

        st.subheader(
            "Emotion Probability per Review"
        )

        st.dataframe(
            result,
            use_container_width=True,
            height=500
        )

# =========================================================
# 6. CUSTOMER SEGMENTATION
# =========================================================

elif menu == "Customer Segmentation":

    st.header("6. Customer Segmentation")

    result = st.session_state.predictions

    if result is None:

        st.warning(
            "Jalankan Emotion Prediction terlebih dahulu."
        )

    else:

        k = st.number_input(
            "Jumlah cluster (K)",
            min_value=2,
            max_value=8,
            value=4
        )

        if st.button(
            "Jalankan K-Means",
            type="primary"
        ):

            with st.spinner(
                "Melakukan clustering..."
            ):

                clustered, profile = perform_clustering(
                    result,
                    k=k
                )

                st.session_state.clustered_data = clustered

                st.session_state.cluster_profile = profile

            st.success(
                "Clustering selesai."
            )

        if st.session_state.clustered_data is not None:

            clustered = (
                st.session_state.clustered_data
            )

            profile = (
                st.session_state.cluster_profile
            )

            st.subheader(
                "Hasil K-Means"
            )

            st.bar_chart(
                clustered["cluster"]
                .value_counts()
                .sort_index()
            )

            st.subheader(
                "Profil Cluster"
            )

            st.dataframe(
                profile,
                use_container_width=True
            )

            st.subheader(
                "Data Hasil Segmentasi"
            )

            st.dataframe(
                clustered,
                use_container_width=True,
                height=500
            )

# =========================================================
# 7. CUSTOMER RETENTION
# =========================================================

elif menu == "Customer Retention":

    st.header("7. Customer Retention")

    clustered = (
        st.session_state.clustered_data
    )

    if clustered is None:

        st.warning(
            "Jalankan Customer Segmentation terlebih dahulu."
        )

    else:

        profile = (
            st.session_state.cluster_profile
        )

        st.subheader(
            "Strategi Customer Retention"
        )

        retention = get_retention_strategy(
            profile
        )

        st.dataframe(
            retention,
            use_container_width=True
        )

        st.divider()

        st.subheader(
            "Prioritas Penanganan"
        )

        priority_order = {
            "Very High": 1,
            "High": 2,
            "Medium": 3,
            "Low": 4
        }

        retention["priority_order"] = (
            retention["prioritas"]
            .map(priority_order)
        )

        retention = (
            retention
            .sort_values("priority_order")
            .drop(columns="priority_order")
        )

        st.dataframe(
            retention,
            use_container_width=True
        )

# =========================================================
# 8. DASHBOARD
# =========================================================

elif menu == "Dashboard":

    st.header("8. Dashboard")

    reviews = st.session_state.reviews

    predictions = st.session_state.predictions

    clustered = st.session_state.clustered_data

    if reviews is None:

        st.info(
            "Dashboard belum memiliki data. "
            "Silakan jalankan Live Scraper."
        )

    else:

        st.subheader(
            "Ringkasan Penelitian"
        )

        total_reviews = len(reviews)

        avg_rating = reviews["rating"].mean()

        if predictions is not None:

            dominant_emotion = (
                predictions["emotion"]
                .value_counts()
                .idxmax()
            )

        else:

            dominant_emotion = "-"

        if clustered is not None:

            total_clusters = (
                clustered["cluster"]
                .nunique()
            )

        else:

            total_clusters = 0

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Total Review",
            total_reviews
        )

        col2.metric(
            "Rata-rata Rating",
            f"{avg_rating:.2f}"
        )

        col3.metric(
            "Emosi Dominan",
            dominant_emotion
        )

        col4.metric(
            "Jumlah Cluster",
            total_clusters
        )

        st.divider()

        if predictions is not None:

            st.subheader(
                "Distribusi Emosi"
            )

            st.bar_chart(
                predictions["emotion"]
                .value_counts()
            )

        if clustered is not None:

            st.subheader(
                "Distribusi Customer Segment"
            )

            st.bar_chart(
                clustered["cluster"]
                .value_counts()
                .sort_index()
            )

        st.divider()

        st.subheader(
            "Alur Analisis"
        )

        st.write(
            """
            Google Play Store Review
            ↓
            Data Understanding
            ↓
            Preprocessing
            ↓
            IndoBERT Emotion Prediction
            ↓
            Emotion Probability
            ↓
            K-Means Customer Segmentation
            ↓
            Customer Retention Strategy
            """
        )

# =========================================================
# 9. SINGLE ANALYSIS
# =========================================================

elif menu == "Single Analysis":

    st.header("9. Single Analysis")

    st.write(
        "Analisis satu ulasan secara langsung."
    )

    review = st.text_area(
        "Masukkan ulasan nasabah",
        height=150,
        placeholder=(
            "Contoh: Aplikasi sangat lambat "
            "dan transaksi sering gagal."
        )
    )

    if st.button(
        "Analisis Ulasan",
        type="primary",
        use_container_width=True
    ):

        if not review.strip():

            st.warning(
                "Masukkan ulasan terlebih dahulu."
            )

        else:

            with st.spinner(
                "Menganalisis ulasan..."
            ):

                processed_text = (
                    preprocess_text(review)
                )

                tokenizer, model = get_model()

                prediction = predict_emotion(
                    processed_text,
                    tokenizer,
                    model
                )

            st.subheader(
                "Hasil Analisis"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Emotion",
                    prediction["emotion"]
                )

            with col2:

                st.metric(
                    "Confidence",
                    f"{prediction['confidence']:.2%}"
                )

            st.divider()

            st.subheader(
                "Emotion Probability"
            )

            probability = prediction[
                "probabilities"
            ]

            for emotion, value in probability.items():

                st.write(
                    f"{emotion}: {value:.2%}"
                )

                st.progress(
                    float(value)
                )

            st.divider()

            st.subheader(
                "Customer Segment & Retention"
            )

            segment = get_single_retention(
                prediction["probabilities"]
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Segment",
                segment["segment"]
            )

            col2.metric(
                "Strategi",
                segment["strategi"]
            )

            col3.metric(
                "Prioritas",
                segment["prioritas"]
            )

            st.info(
                segment["rekomendasi"]
            )
