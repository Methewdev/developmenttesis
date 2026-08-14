# =========================================================
# CONFIGURATION
# =========================================================

MODEL_PATH = "models/indobert_emotion/checkpoint"

MODEL_NAME = "IndoBERT"

RANDOM_STATE = 42

N_CLUSTERS = 4

MAX_LENGTH = 128

# ---------------------------------------------------------
# LABEL
# ---------------------------------------------------------

EMOTION_LABELS = [
    "Senang",
    "Sedih",
    "Frustrasi",
    "Netral"
]

LABEL_MAPPING = {
    0: "Senang",
    1: "Sedih",
    2: "Frustrasi",
    3: "Netral"
}

# ---------------------------------------------------------
# CLUSTER PROFILE
# ---------------------------------------------------------

CLUSTER_NAMES = {
    0: "Puas",
    1: "Sedih Murni",
    2: "Tidak Puas Campuran",
    3: "Frustrasi"
}

# ---------------------------------------------------------
# RETENTION
# ---------------------------------------------------------

RETENTION_STRATEGY = {

    "Puas": {
        "prioritas": "Low",
        "strategi": "Loyalty & Engagement",
        "eksekutor": "Marketing / CRM",
        "kanal": "In-App / Push Notification",
        "kpi": "Retention Rate"
    },

    "Sedih Murni": {
        "prioritas": "High",
        "strategi": "Service Recovery",
        "eksekutor": "Customer Service",
        "kanal": "Contact Center / In-App",
        "kpi": "Resolution Rate"
    },

    "Tidak Puas Campuran": {
        "prioritas": "High",
        "strategi": "Improvement & Follow-up",
        "eksekutor": "Product / Customer Experience",
        "kanal": "In-App / Email",
        "kpi": "Complaint Reduction"
    },

    "Frustrasi": {
        "prioritas": "Very High",
        "strategi": "Immediate Service Recovery",
        "eksekutor": "Customer Service / Product",
        "kanal": "Contact Center / In-App",
        "kpi": "Recovery Rate"
    }
}
