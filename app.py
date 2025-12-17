import streamlit as st
import pickle
import numpy as np

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

st.set_page_config(page_title="ML Prediction App", layout="centered")
st.title("🤖 Machine Learning Prediction")

# Example inputs (change based on your dataset)
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")

if st.button("Predict"):
    data = np.array([[feature1, feature2]])
    result = model.predict(data)
    st.success(f"Prediction: {result[0]}")
