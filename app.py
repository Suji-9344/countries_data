import streamlit as st
import pickle

st.title("Countries Clustered Data")

with open("countries_clustered.pkl", "rb") as f:
    data = pickle.load(f)

st.success("PKL file loaded successfully ✅")
st.dataframe(data)
