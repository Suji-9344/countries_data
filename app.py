import streamlit as st
import pickle
import os
import numpy as np

st.set_page_config(page_title="ML App", layout="centered")

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")

    if not os.path.exists(model_path):
        st.error("❌ model.pkl file not found. Please upload it to GitHub.")
        st.stop()

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# ---------- UI ----------
st.title("🤖 Prediction App")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Predict"):
    data = np.array([[f1, f2]])
    prediction = model.predict(data)
    st.success(f"Prediction: {prediction[0]}")

