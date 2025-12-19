import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster

st.set_page_config(page_title="Interactive Hierarchical Clustering", layout="centered")
st.title("🌍 Interactive Hierarchical Clustering of Countries (No Plot)")

st.write("Enter country names and numeric values for clustering.")

# ------------------- USER INPUT -------------------
st.subheader("Add Country Data")

# Initialize empty list to store countries
if "countries_data" not in st.session_state:
    st.session_state.countries_data = []

with st.form("country_form", clear_on_submit=True):
    country_name = st.text_input("Country Name")
    gdp = st.number_input("GDP (in USD)", min_value=0.0, value=0.0)
    life_exp = st.number_input("Life Expectancy (years)", min_value=0.0, value=70.0)
    population = st.number_input("Population", min_value=0, value=1000000)
    hdi = st.number_input("HDI (0-1)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    
    submitted = st.form_submit_button("Add Country")
    
    if submitted:
        st.session_state.countries_data.append({
            "Country": country_name,
            "GDP": gdp,
            "LifeExpectancy": life_exp,
            "Population": population,
            "HDI": hdi
        })
        st.success(f"✅ {country_name} added to dataset.")

# ------------------- SHOW CURRENT DATA -------------------
if st.session_state.countries_data:
    st.subheader("Current Dataset")
    df = pd.DataFrame(st.session_state.countries_data)
    st.dataframe(df)
    
    # ------------------- CLUSTERING -------------------
    numeric_cols = ["GDP", "LifeExpectancy", "Population", "HDI"]
    X = df[numeric_cols]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform hierarchical clustering
    Z = linkage(X_scaled, method='ward')

    # Select number of clusters
    n_clusters = st.slider("Select number of clusters", 2, 10, 3)

    # Assign cluster labels
    df['Cluster'] = fcluster(Z, n_clusters, criterion='maxclust')

    st.subheader("📊 Countries with Cluster Labels")
    st.dataframe(df)

    # Download option
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Clustered CSV", data=csv, file_name="countries_clustered.csv", mime="text/csv")
