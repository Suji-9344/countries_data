import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff

st.set_page_config(page_title="Countries Hierarchical Clustering", layout="wide")
st.title("🌍 Hierarchical Clustering of Countries")

st.write("This app performs hierarchical clustering on a default countries dataset.")

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

# ------------------- CLUSTERING -------------------
numeric_cols = df.select_dtypes(include='number').columns.tolist()
X = df[numeric_cols]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method='ward')

# Dendrogram
st.subheader("📈 Dendrogram")
labels = df['Country'].tolist()
fig = ff.create_dendrogram(X_scaled, labels=labels, orientation='top', linkagefun=lambda x: Z)
fig.update_layout(width=900, height=500)
st.plotly_chart(fig)

# Number of clusters
n_clusters = st.slider("Select number of clusters", 2, 10, 3)

# Assign cluster labels
df['Cluster'] = fcluster(Z, n_clusters, criterion='maxclust')

st.subheader("📊 Clustered Countries")
st.dataframe(df)

# Download clustered CSV
csv = df.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download Clustered CSV", data=csv, file_name="countries_clustered.csv", mime="text/csv")
