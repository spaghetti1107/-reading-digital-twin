import streamlit as st
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Reading Digital Twin",
    page_icon="👁️",
    layout="wide"
)

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("ETDD70_features.csv")

X = df.drop(
    columns=["subject_id", "class_id", "label"]
)

y = df["class_id"]

# ============================================================
# READING SPEED
# ============================================================

# Reconstruct reading speed if CSV is available
try:
    reading_speed_df = pd.read_csv("reading_speed.csv")
except:
    reading_speed_df = None


# ============================================================
# TRAIN FINAL SVM
# ============================================================

final_svm = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),

    (
        "scaler",
        StandardScaler()
    ),

    (
        "classifier",
        SVC(
            kernel="rbf",
            probability=True,
            random_state=42
        )
    )
])

final_svm.fit(X, y)


# ============================================================
# TITLE
# ============================================================

st.title("👁️ Reading Digital Twin")

st.markdown(
    """
    ### Eye-Tracking Based Reading Profile

    This research prototype uses eye-tracking biomarkers
    to generate an individualized reading profile.
    """
)

st.divider()


# ============================================================
# CHILD SELECTION
# ============================================================

child_ids = sorted(
    df["subject_id"].unique()
)

selected_child = st.selectbox(
    "Select Child",
    child_ids
)

child = df[
    df["subject_id"] == selected_child
].iloc[0]

idx = df.index[
    df["subject_id"] == selected_child
][0]

child_X = X.iloc[[idx]]


# ============================================================
# PREDICTION
# ============================================================

prediction = final_svm.predict(
    child_X
)[0]

probability = final_svm.predict_proba(
    child_X
)[0, 1]


# ============================================================
# CLASSIFICATION CARDS
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "ML Classification",
        "Dyslexic" if prediction == 1
        else "Non-dyslexic"
    )

with col2:

    st.metric(
        "Model-estimated dyslexic-class probability",
        f"{probability * 100:.1f}%"
    )

with col3:

    st.metric(
        "Dataset",
        "ETDD70"
    )


st.divider()


# ============================================================
# BIOMARKER TABLE
# ============================================================

st.header("📊 Eye-Tracking Biomarkers")

biomarker_rows = []

for task in ["T1", "T4", "T5"]:

    task_name = {
        "T1": "T1 — Syllables",
        "T4": "T4 — Meaningful Text",
        "T5": "T5 — Pseudo Text"
    }[task]

    biomarker_rows.append({

        "Task":
            task_name,

        "Mean Fixation Duration":
            child[f"mean_fix_dur_trial_{task}"],

        "Fixation Count":
            child[f"n_fix_trial_{task}"],

        "Regression Count":
            child[f"n_regress_trial_{task}"],

        "Progress/Regression Ratio":
            child[f"ratio_progress_regress_trial_{task}"],

        "Saccade Amplitude":
            child[f"mean_sacc_ampl_trial_{task}"],

        "Revisits":
            child[f"n_revisits_aoi_{task}"],

        "Dwell Time":
            child[f"dwell_time_trial_{task}"],

        "Transit Count":
            child[f"n_transit_trial_{task}"]
    })


biomarker_df = pd.DataFrame(
    biomarker_rows
)

st.dataframe(
    biomarker_df,
    use_container_width=True
)


# ============================================================
# READING SPEED
# ============================================================

st.header("📖 Reading Speed")

if reading_speed_df is not None:

    speed = reading_speed_df[
        reading_speed_df["subject_id"]
        == selected_child
    ].copy()

    if len(speed) > 0:

        speed_display = speed[
            [
                "task",
                "reading_speed_wpm"
            ]
        ].copy()

        speed_display.columns = [
            "Task",
            "Reading Speed (WPM)"
        ]

        st.bar_chart(
            speed_display.set_index("Task")
        )

        st.dataframe(
            speed_display,
            use_container_width=True
        )

    else:

        st.info(
            "Reading-speed data not available for this child."
        )

else:

    st.info(
        "reading_speed.csv not found."
    )


# ============================================================
# BIOMARKER PROFILE
# ============================================================

st.header("🧠 Reading Profile")

observations = []
recommendations = []


# ------------------------------------------------------------
# Pseudo-text speed
# ------------------------------------------------------------

if reading_speed_df is not None:

    speed = reading_speed_df[
        reading_speed_df["subject_id"]
        == selected_child
    ]

    try:

        t1 = float(
            speed.loc[
                speed["task"] == "T1",
                "reading_speed_wpm"
            ].iloc[0]
        )

        t4 = float(
            speed.loc[
                speed["task"] == "T4",
                "reading_speed_wpm"
            ].iloc[0]
        )

        t5 = float(
            speed.loc[
                speed["task"] == "T5",
                "reading_speed_wpm"
            ].iloc[0]
        )

        mean_other = (t1 + t4) / 2

        if t5 < 0.75 * mean_other:

            observations.append(
                "Pseudo-text reading speed is substantially "
                "lower than T1/T4 reading speed."
            )

            recommendations.append(
                "Use graded unfamiliar-word and decoding "
                "fluency exercises."
            )

    except:

        pass


# ------------------------------------------------------------
# Fixations
# ------------------------------------------------------------

for task in ["T1", "T4", "T5"]:

    feature = (
        f"mean_fix_dur_trial_{task}"
    )

    if child[feature] > df[feature].quantile(0.75):

        observations.append(
            f"{task} shows relatively high fixation duration."
        )

        recommendations.append(
            "Use short repeated-reading exercises to "
            "develop reading fluency."
        )


# ------------------------------------------------------------
# Regressions
# ------------------------------------------------------------

for task in ["T1", "T4", "T5"]:

    feature = (
        f"n_regress_trial_{task}"
    )

    if child[feature] > df[feature].quantile(0.75):

        observations.append(
            f"{task} shows relatively elevated regression frequency."
        )

        recommendations.append(
            "Use guided reading and phrase-level tracking."
        )


# ------------------------------------------------------------
# Default
# ------------------------------------------------------------

if len(recommendations) == 0:

    observations.append(
        "No strongly elevated fixation, regression, "
        "or revisit measure was detected relative "
        "to the ETDD70 reference population."
    )

    recommendations.append(
        "Continue regular reading-fluency practice and "
        "monitor changes over repeated assessments."
    )


# Remove duplicates

observations = list(
    dict.fromkeys(observations)
)

recommendations = list(
    dict.fromkeys(recommendations)
)


# ============================================================
# DISPLAY PROFILE
# ============================================================

for observation in observations:

    st.info(
        observation
    )


st.subheader(
    "Suggested Intervention"
)

for i, recommendation in enumerate(
    recommendations,
    1
):

    st.write(
        f"**{i}.** {recommendation}"
    )


# ============================================================
# IMPORTANT NOTE
# ============================================================

st.divider()

st.warning(
    """
    Research prototype only. Model outputs represent
    machine-learning estimates from eye-tracking data
    and should not be interpreted as a clinical diagnosis
    or medical treatment recommendation.
    """
)
