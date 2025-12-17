import streamlit as st
import pickle
import os
import numpy as np

st.set_page_config(page_title="ML App", layout="centered")

# ---------- AUTO FIND MODEL ----------
@st.cache_resource
def load_model():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    found_path = None
    for root, dirs, files in os.walk(root_dir):
        if "model.pkl" in files:
            found_path = os.path.join(root, "model.pkl")
            break

    if found_path is None:
        st.error("❌ model.pkl NOT FOUND anywhere in this GitHub repo.")
        st.write("🔍 Searched directory:", root_dir)
        st.stop()

    st.success(f"✅ model.pkl found at: {found_path}")

    with open(found_path, "rb") as f:
        return pickle.load(f)

model = load_model()

# ---------- UI ----------
st.title("🤖 Prediction App")

f1 = st.number_input("Feature 1")
f2 = st.number_input("Feature 2")

if st.button("Predict"):
    result = model.predict([[f1, f2]])
    st.success(f"Prediction: {result[0]}")
    import streamlit as st
import pickle
import pandas as pd

st.title("Countries Clustered Data")

with open("countries_clustered.pkl", "rb") as f:
    data = pickle.load(f)

st.success("PKL file loaded successfully ✅")
st.dataframe(data)
