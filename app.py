import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import matplotlib.pyplot as plt

st.set_page_config(page_title="Countries Hierarchical Clustering", layout="wide")
st.title("🌍 Hierarchical Clustering of Countries")

st.write("Upload a CSV file with countries and numeric features (e.g., GDP, Population, Life Expectancy, HDI).")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV file loaded successfully!")
    
    # Show first 5 rows
    with st.expander("📊 Preview Dataset"):
        st.dataframe(df.head())

    # Check numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) < 1:
        st.error("❌ No numeric columns found for clustering!")
    else:
        st.write(f"Using numeric columns for clustering: {numeric_cols}")

        # Scale numeric features
        X = df[numeric_cols]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Linkage matrix
        Z = linkage(X_scaled, method='ward')

        # Dendrogram
        st.subheader("📈 Dendrogram")
        plt.figure(figsize=(10, 5))
        dendrogram(Z, labels=df['Country'].values if 'Country' in df.columns else None, leaf_rotation=90)
        st.pyplot(plt.gcf())
        plt.clf()

        # Number of clusters
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)

        # Assign cluster labels
        df['Cluster'] = fcluster(Z, n_clusters, criterion='maxclust')

        st.subheader("📊 Countries with Cluster Labels")
        st.dataframe(df)

        # Optional: Download clustered dataset
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Clustered CSV", data=csv, file_name="countries_clustered.csv", mime="text/csv")
