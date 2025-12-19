import streamlit as st
import pickle

st.title("Countries Clustered Data")

with open("countries_clustered.pkl", "rb") as f:
    data = pickle.load(f)

st.success("PKL file loaded successfully ✅")
st.dataframe(data)
import streamlit as st
import pickle
import os

st.set_page_config(page_title="Countries Cluster App", layout="centered")

st.title("🌍 Countries Clustered ML App")

# -----------------------------------
# LOAD PKL FILE SAFELY
# -----------------------------------
@st.cache_resource
def load_pkl():
    file_name = None

    # Check possible file names
    if os.path.exists("countries_clustered.pkl"):
        file_name = "countries_clustered.pkl"
    elif os.path.exists("countries_clustered (1).pkl"):
        file_name = "countries_clustered (1).pkl"

    # If NOT found
    if file_name is None:
        st.error("❌ PKL file not found in repo")
        st.stop()

    # Load file
    with open(file_name, "rb") as f:
        data = p

