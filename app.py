import streamlit as st
import pickle
import os
import numpy as np

st.set_page_config(page_title="ML App", layout="centered")

@st.cache_resource
def load_model():
    base_dir = os.path.dirname(__file__)
    model_path = os.path.join(base_dir, "..", "model.pkl")

    if not os.path.exists(model_path):
        st.error(f"❌ model.pkl not found at {model_path}")
        st.stop()

    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model()

st.title("🤖 Prediction App")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Predict"):
    result = model.predict([[f1, f2]])
    st.success(f"Prediction: {result[0]}")
