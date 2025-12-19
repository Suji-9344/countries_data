import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

st.set_page_config(page_title="Countries Hierarchical Clustering", layout="centered")
st.title("🌍 Hierarchical Clustering of Countries (No Plot)")

st.write("This app performs hierarchical clustering on a default countries dataset without plotting dendrograms.")

# ------------------- DEFAULT DATASET -------------------
data = {
    "Country": ["India", "USA", "China", "UK", "Japan", "Germany", "Brazil", "Australia", "Canada", "Russia"],
    "GDP": [2200, 65000, 12000, 43000, 42000, 46000, 9000, 54000, 48000, 11000],
    "LifeExpectancy": [70.8, 79.1, 76.5, 80.5, 84.6, 81.2, 75.5, 82.8, 82.3, 72.6],
    "Population": [1400000000, 331000000, 1440000000, 68000000, 125000000, 83000000, 212000000, 25000000, 38000000, 146000000],
    "HDI": [0.63, 0.92, 0.74, 0.90, 0.91, 0.94, 0.76, 0.94, 0.93, 0.82]
}

df = pd.DataFrame(data)

# ------------------- SHOW DATASET ON DEMAND -------------------
if st.button("📊 Show Dataset"):
    st.subheader("Default Countries Dataset")
    st.dataframe(df)

# ----------
