import streamlit as st
import numpy as np
import pandas as pd
import joblib
import shap
import lime
import lime.lime_tabular
import tensorflow as tf
import matplotlib.pyplot as plt
from io import BytesIO
from fpdf import FPDF
from fpdf.enums import XPos, YPos
import re, unicodedata
from datetime import datetime
import os

BASE_DIR = os.path.dirname(os.path.abspath(r"C:\Users\kinsh\OneDrive\Desktop\Heart Pre Final"))


scaler = joblib.load(os.path.join(BASE_DIR, "heart_scaler.pkl"))
xgb_model = joblib.load(os.path.join(BASE_DIR, "heart_xgb_tuned.pkl"))
att_model = tf.keras.models.load_model(os.path.join(BASE_DIR, "heart_attention_regressor.h5"), compile=False)


# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(page_title="Heart Disease Risk Predictor", page_icon="❤️", layout="centered")
st.title("❤️ Heart Disease Risk Predictor")
st.markdown("### Predict and interpret your heart disease risk using explainable AI")

# ==========================
# SIDEBAR INPUT FORM
# ==========================
st.sidebar.header("🧍 Patient Information")

def user_input():
    age = st.sidebar.slider("Age", 20, 90, 50)
    sex = st.sidebar.selectbox("Sex (1=Male, 0=Female)", [1, 0])
    cp = st.sidebar.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    trestbps = st.sidebar.slider("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.sidebar.slider("Serum Cholesterol (mg/dl)", 100, 600, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", [1, 0])
    restecg = st.sidebar.selectbox("Resting ECG Results (0–2)", [0, 1, 2])
    thalach = st.sidebar.slider("Max Heart Rate Achieved", 70, 220, 150)
    exang = st.sidebar.selectbox("Exercise-Induced Angina (1=True, 0=False)", [1, 0])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.selectbox("Slope of Peak Exercise ST Segment (0–2)", [0, 1, 2])
    ca = st.sidebar.slider("Number of Major Vessels (0–4)", 0, 4, 0)
    thal = st.sidebar.selectbox("Thalassemia (0–3)", [0, 1, 2, 3])

    data = {
        'age': age, 'sex': sex, 'cp': cp, 'trestbps': trestbps, 'chol': chol,
        'fbs': fbs, 'restecg': restecg, 'thalach': thalach, 'exang': exang,
        'oldpeak': oldpeak, 'slope': slope, 'ca': ca, 'thal': thal
    }
    return pd.DataFrame(data, index=[0])

input_df = user_input()

# ==========================
# STEP 1 — PREDICTION
# ==========================
st.subheader("🧮 Model Prediction")

X_scaled = scaler.transform(input_df)
proba = xgb_model.predict_proba(X_scaled)[:, 1][0]
pred_risk = att_model.predict(np.array([[proba]])).flatten()[0]

def interpret_risk(prob, prox):
    if prob < 0.3:
        return ("Low Risk", "Your heart appears healthy. Maintain healthy habits.")
    elif 0.3 <= prob < 0.7:
        if prox < 0.5:
            return ("Moderate Risk (Stable)", "Mild signs of heart strain. Monitor health regularly.")
        else:
            return ("Moderate Risk (Rising)", "Risk rising - consider consulting a cardiologist.")
    else:
        return ("High Risk", "High likelihood of heart disease. Seek medical attention soon.")

level, message = interpret_risk(proba, pred_risk)
st.metric(label="Predicted Probability", value=f"{proba*100:.1f}%")
st.metric(label="Risk Proximity", value=f"{pred_risk:.2f}")
st.markdown(f"### {level}")
st.info(message)

# ==========================
# STEP 2 — SHAP EXPLAINABILITY
# ==========================
st.subheader("🧠 SHAP Feature Importance")

try:
    explainer = shap.Explainer(xgb_model.predict_proba, X_scaled)
    shap_values = explainer(X_scaled)
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(shap.Explanation(
        values=shap_values.values[0],
        base_values=shap_values.base_values[0],
        feature_names=input_df.columns,
        data=input_df.values[0]
    ))
    st.pyplot(fig)
except Exception as e:
    st.warning(f"⚠️ SHAP explanation skipped: {e}")

# ==========================
# STEP 3 — LIME EXPLANATION
# ==========================
st.subheader("🌿 LIME Local Explanation")

explainer_lime = lime.lime_tabular.LimeTabularExplainer(
    training_data=scaler.transform(input_df.values),
    feature_names=input_df.columns.tolist(),
    class_names=['No Disease', 'Disease'],
    mode='classification'
)
lime_exp = explainer_lime.explain_instance(
    X_scaled[0],
    xgb_model.predict_proba,
    num_features=5
)
lime_html = lime_exp.as_html()
st.components.v1.html(lime_html, height=400, scrolling=True)

# ==========================
# STEP 4 — GENERATE REPORT
# ==========================
st.subheader("📄 Generate Clinical Report")

report_data = {
    "Predicted Probability": f"{proba*100:.2f}%",
    "Risk Proximity": f"{pred_risk:.2f}",
    "Risk Level": level,
    "Recommendation": message
}
report_df = pd.DataFrame(report_data.items(), columns=["Parameter", "Value"])
csv = report_df.to_csv(index=False).encode('utf-8')

# --- Clean Text Helper (fix Unicode issues)
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.replace("—", "-").replace("–", "-").replace("…", "...")
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\x00-\xFF]", "", text)
    return text

# --- Create PDF (Unicode-safe, warning-free)
def create_pdf(report_df, input_df):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)

    pdf.cell(200, 10,
             text="Heart Disease Risk Report",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")
    pdf.ln(5)

    for _, row in report_df.iterrows():
        line = f"{clean_text(row['Parameter'])}: {clean_text(row['Value'])}"
        pdf.cell(200, 10,
                 text=line,
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(5)
    pdf.cell(200, 10,
             text="Patient Input Summary:",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    for col, val in input_df.iloc[0].items():
        pdf.cell(200, 8,
                 text=f"{clean_text(col)}: {clean_text(val)}",
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    pdf.ln(5)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pdf.set_font("Helvetica", size=10)
    pdf.cell(200, 10,
             text=f"Report generated on {timestamp}",
             new_x=XPos.LMARGIN, new_y=YPos.NEXT, align="C")

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

pdf_bytes = create_pdf(report_df, input_df)

# --- Side-by-side download buttons
col1, col2 = st.columns(2)
with col1:
    st.download_button("⬇️ Download as CSV", data=csv, file_name="heart_risk_report.csv", mime="text/csv")
with col2:
    st.download_button("📄 Download as PDF", data=pdf_bytes, file_name="heart_risk_report.pdf", mime="application/pdf")

st.success("✅ Your personalized heart disease risk report is ready for download!")

# ==========================
# FOOTER
# ==========================
st.markdown("---")
st.markdown("🔬 Developed using **XGBoost + Attention Regression + SHAP + LIME** for transparent cardiology AI.")
# ==========================
# 🤖 CONTEXT-AWARE MEDICAL CHATBOT
# ==========================
# ==========================
# STEP 5 — AI MEDICAL CHATBOT
# ==========================

# ==========================
# 🤖 CONTEXT-AWARE MEDICAL CHATBOT (WITH MEMORY)
# ==========================

# ==========================
# STEP 5 — AI MEDICAL CHATBOT
# ==========================
# ==========================
# STEP 5 — OFFLINE CONVERSATIONAL MEDICAL CHATBOT
# ==========================
import numpy as np
import os
from sentence_transformers import SentenceTransformer, util

st.markdown("---")
st.subheader("💬 Ask the AI Medical Assistant (Offline & Context-Aware)")

# --- Initialize chat history safely ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ==========================
# 🧠 Preloaded medical knowledge base
# ==========================
medical_knowledge = {
    "cholesterol": "A healthy total cholesterol level is below 200 mg/dL. HDL (good cholesterol) should be above 40 mg/dL for men and 50 mg/dL for women.",
    "blood pressure": "Normal resting blood pressure is around 120/80 mmHg. High blood pressure increases heart disease risk.",
    "heart disease": "Heart disease happens when coronary arteries become narrowed, limiting blood flow. Major risk factors include smoking, hypertension, and high cholesterol.",
    "exercise": "Engaging in 150 minutes of moderate physical activity per week improves heart health and reduces hypertension risk.",
    "diet": "A balanced diet with fruits, vegetables, lean protein, and minimal salt and sugar helps reduce heart disease risk.",
    "symptoms": "Chest pain, fatigue, and shortness of breath can be early signs of heart disease. Persistent symptoms should be discussed with a doctor.",
    "shap": "SHAP values explain how each input factor (like age, blood pressure, cholesterol) affected your AI model’s prediction.",
    "lime": "LIME helps interpret the model by showing how individual features contributed to your predicted heart risk."
}

# ==========================
# ⚙️ Helper Functions
# ==========================
def to_scalar(value):
    """Convert arrays/lists/tensors to simple float values."""
    if isinstance(value, (list, np.ndarray)):
        try:
            return float(np.mean(value))
        except Exception:
            return 0.0
    try:
        return float(value)
    except Exception:
        return 0.0


@st.cache_resource
def load_embedder():
    """Load lightweight semantic model."""
    return SentenceTransformer("all-MiniLM-L6-v2")


embedder = load_embedder()

# Precompute embeddings for knowledge base
keys = list(medical_knowledge.keys())
values = list(medical_knowledge.values())
knowledge_embeddings = embedder.encode(keys, convert_to_tensor=True)

# ==========================
# 💬 Chat Input
# ==========================
user_query = st.text_input("Ask about your results, SHAP, LIME, or general heart health:")

if user_query:
    with st.spinner("💭 Thinking..."):
        try:
            # Convert predictions safely
            proba_scalar = to_scalar(proba)
            pred_risk_scalar = to_scalar(pred_risk)

            # --- Build context string from chat history ---
            history_context = ""
            if len(st.session_state.chat_history) > 0:
                history_context = " ".join(
                    [msg["content"] for msg in st.session_state.chat_history[-4:]]
                )  # last 4 exchanges only

            # --- Combine history + new query for context-aware matching ---
            full_query = history_context + " " + user_query

            # --- Embed query ---
            query_emb = embedder.encode(full_query, convert_to_tensor=True)
            scores = util.pytorch_cos_sim(query_emb, knowledge_embeddings)[0]
            best_match_idx = int(scores.argmax())

            # --- Retrieve best response ---
            base_response = values[best_match_idx]

            # --- Add personalization based on model results ---
            context_info = (
                f"Based on your current model insights:\n"
                f"- Predicted Heart Disease Probability: {proba_scalar*100:.1f}%\n"
                f"- Risk Proximity: {pred_risk_scalar:.2f}\n"
                f"- Risk Level: {level}\n\n"
            )

            # --- Make it conversational ---
            if "follow" in user_query.lower() or "more" in user_query.lower():
                response = f"You asked for more information. {base_response}"
            elif "thank" in user_query.lower():
                response = "You're welcome! Always take care of your heart and health. 💓"
            elif "hello" in user_query.lower() or "hi" in user_query.lower():
                response = "Hello there! 👋 How can I assist you with your heart health today?"
            else:
                response = f"{context_info}{base_response}"

            # --- Save and display ---
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            st.session_state.chat_history.append({"role": "assistant", "content": response})

            st.success(f"**🩺 Assistant:** {response}")

        except Exception as e:
            st.error(f"⚠️ Chatbot error: {e}")

# ==========================
# 🧾 Display Chat History
# ==========================
if st.session_state.chat_history:
    st.markdown("### 🗨️ Conversation")
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**👤 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 Assistant:** {msg['content']}")

# ==========================
# 🔁 Clear Chat Button
# ==========================
if st.button("🧹 Clear Chat"):
    st.session_state.chat_history = []
    st.success("Chat cleared! Ready for a new session.")


