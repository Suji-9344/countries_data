import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster
import plotly.figure_factory as ff

st.set_page_config(page_title="Countries Hierarchical Clustering", layout="wide")
st.title("🌍 Hierarchical Clustering of Countries")

st.write("Upload a CSV with numeric features (e.g., GDP, Population, HDI). Column 'Country' is optional for labels.")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ CSV file loaded successfully!")
    
    # Preview dataset
    with st.expander("📊 Preview Dataset"):
        st.dataframe(df.head())

    # Identify numeric columns
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if len(numeric_cols) < 1:
        st.error("❌ No numeric columns found for clustering!")
    else:
        st.write(f"Using numeric columns: {numeric_cols}")
        X = df[numeric_cols]

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform hierarchical clustering
        Z = linkage(X_scaled, method='ward')

        # Plot dendrogram with Plotly
        st.subheader("📈 Dendrogram")
        labels = df['Country'].tolist() if 'Country' in df.columns else None
        fig = ff.create_dendrogram(X_scaled, labels=labels, orientation='top', linkagefun=lambda x: Z)
        fig.update_layout(width=900, height=500)
        st.plotly_chart(fig)

        # Choose number of clusters
        n_clusters = st.slider("Select number of clusters", 2, 10, 3)

        # Assign cluster labels
        df['Cluster'] = fcluster(Z, n_clusters, criterion='maxclust')
        st.subheader("📊 Clustered Countries")
        st.dataframe(df)

        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Clustered CSV", data=csv, file_name="countries_clustered.csv", mime="text/csv")
