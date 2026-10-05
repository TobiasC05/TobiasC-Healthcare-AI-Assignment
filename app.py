import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# ------------------------------------------------------------
# AI-Powered Symptom Checker and Triage Demo
# Educational prototype only - NOT a medical device.
# ------------------------------------------------------------

st.set_page_config(
    page_title="AI Symptom Triage Demo",
    page_icon="🩺",
    layout="centered"
)

# Small demonstration dataset.
# In a real healthcare system, training data would need to be
# clinically validated, representative, securely governed, and
# reviewed for bias and safety.
TRAINING_EXAMPLES = [
    # Emergency
    ("crushing chest pain shortness of breath sweating nausea", "Emergency", "Cardiac warning pattern"),
    ("chest pressure trouble breathing pain going to arm", "Emergency", "Cardiac warning pattern"),
    ("face drooping arm weakness slurred speech sudden", "Emergency", "Stroke warning pattern"),
    ("sudden weakness one side cannot speak", "Emergency", "Stroke warning pattern"),
    ("severe bleeding will not stop fainting", "Emergency", "Severe bleeding"),
    ("unconscious not responding difficulty breathing", "Emergency", "Loss of consciousness / breathing emergency"),
    ("severe allergic reaction swelling lips tongue cannot breathe", "Emergency", "Severe allergic reaction"),
    ("seizure lasting several minutes not waking up", "Emergency", "Neurologic emergency"),

    # Urgent
    ("high fever persistent cough weakness body aches", "Urgent", "Respiratory infection pattern"),
    ("fever cough shortness of breath getting worse", "Urgent", "Respiratory infection pattern"),
    ("severe headache fever neck stiffness", "Urgent", "Neurologic / infection warning pattern"),
    ("abdominal pain vomiting cannot keep fluids down", "Urgent", "Gastrointestinal illness pattern"),
    ("painful urination fever back pain", "Urgent", "Urinary infection pattern"),
    ("deep cut redness swelling pus fever", "Urgent", "Wound / infection pattern"),
    ("persistent vomiting dizziness dehydration", "Urgent", "Dehydration risk"),
    ("wheezing cough breathing harder than normal", "Urgent", "Respiratory warning pattern"),

    # Self-care / non-urgent
    ("runny nose mild cough sore throat no fever", "Self-care", "Common cold pattern"),
    ("mild headache after long day better with rest", "Self-care", "Tension headache pattern"),
    ("itchy eyes sneezing runny nose seasonal", "Self-care", "Allergy pattern"),
    ("mild upset stomach after eating no severe pain", "Self-care", "Mild digestive pattern"),
    ("small bruise mild soreness after bump", "Self-care", "Minor injury pattern"),
    ("mild muscle soreness after exercise", "Self-care", "Muscle strain / exercise soreness"),
    ("mild sore throat congestion improving", "Self-care", "Upper respiratory pattern"),
    ("minor cough no fever breathing normally", "Self-care", "Mild respiratory pattern"),
]

texts = [x[0] for x in TRAINING_EXAMPLES]
triage_labels = [x[1] for x in TRAINING_EXAMPLES]
condition_labels = [x[2] for x in TRAINING_EXAMPLES]

# Two simple machine-learning pipelines.
# TF-IDF converts symptom text into numerical features.
# Logistic Regression learns patterns associated with each label.
triage_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

condition_model = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

triage_model.fit(texts, triage_labels)
condition_model.fit(texts, condition_labels)

# Red-flag rules are intentionally placed before the ML output.
# This is a safety layer so obvious emergency phrases are not
# downgraded by the tiny demonstration model.
RED_FLAGS = [
    r"\bchest pain\b.*\b(shortness of breath|trouble breathing|sweating)\b",
    r"\b(face droop|face drooping|slurred speech|one side weak|one-sided weakness)\b",
    r"\b(unconscious|not responding|cannot wake|can't wake)\b",
    r"\b(severe bleeding|bleeding will not stop|won't stop bleeding)\b",
    r"\b(swelling (of )?(lips|tongue)|cannot breathe|can't breathe)\b",
    r"\b(seizure)\b.*\b(5 minutes|five minutes|not waking|not responding)\b",
]

def has_red_flag(text: str) -> bool:
    """Return True if the text contains a predefined emergency warning pattern."""
    normalized = text.lower().strip()
    return any(re.search(pattern, normalized) for pattern in RED_FLAGS)

def predict(text: str):
    """
    Predict a triage level and a broad symptom pattern.
    Returns labels plus model confidence estimates.
    """
    if has_red_flag(text):
        return {
            "triage": "Emergency",
            "triage_confidence": 1.0,
            "pattern": "Emergency warning pattern",
            "pattern_confidence": 1.0,
            "red_flag": True,
        }

    triage = triage_model.predict([text])[0]
    triage_probs = triage_model.predict_proba([text])[0]
    triage_conf = float(max(triage_probs))

    pattern = condition_model.predict([text])[0]
    condition_probs = condition_model.predict_proba([text])[0]
    condition_conf = float(max(condition_probs))

    return {
        "triage": triage,
        "triage_confidence": triage_conf,
        "pattern": pattern,
        "pattern_confidence": condition_conf,
        "red_flag": False,
    }

# ------------------------- UI -------------------------

st.title("🩺 AI-Powered Symptom Checker & Triage Demo")
st.caption("Course project prototype using NLP + machine learning")

st.warning(
    "Educational demonstration only. This tool does not diagnose disease and is "
    "not a substitute for a doctor or emergency services."
)

symptoms = st.text_area(
    "Describe symptoms in your own words:",
    placeholder="Example: I have a high fever, cough, and I feel very weak.",
    height=130
)

if st.button("Analyze Symptoms", type="primary"):
    if not symptoms.strip():
        st.info("Please enter a symptom description first.")
    else:
        result = predict(symptoms)

        st.subheader("Demo Result")
        st.write(f"**Suggested triage level:** {result['triage']}")
        st.write(f"**Possible symptom pattern:** {result['pattern']}")

        if result["red_flag"]:
            st.error(
                "The text contains an emergency warning pattern. "
                "Call 911 (U.S.) or your local emergency number now."
            )
        elif result["triage"] == "Urgent":
            st.warning(
                "Consider prompt evaluation by a healthcare professional, "
                "especially if symptoms are worsening or you are concerned."
            )
        else:
            st.info(
                "This demo classified the description as lower urgency, but "
                "seek medical care if symptoms worsen, persist, or concern you."
            )

        with st.expander("How the AI works"):
            st.write(
                "The app converts symptom text into TF-IDF features and uses "
                "Logistic Regression to classify triage level and a broad symptom "
                "pattern. A separate red-flag rule layer checks for several obvious "
                "emergency phrases before showing the machine-learning result."
            )
            if not result["red_flag"]:
                st.write(
                    f"Model confidence (triage): {result['triage_confidence']:.2f}"
                )
                st.write(
                    f"Model confidence (pattern): {result['pattern_confidence']:.2f}"
                )

st.divider()
st.markdown(
    """
### Limitations
- The training dataset is tiny and manually created for a class demonstration.
- The model is **not clinically validated** and should never be used for real diagnosis.
- Real medical AI requires representative data, privacy protection, bias testing,
  clinical evaluation, regulatory review, and ongoing monitoring.
"""
)
