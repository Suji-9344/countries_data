import streamlit as st

st.title('Country Clustering Dashboard')
import pandas as pd

df = pd.read_csv('/content/countries_clustered.csv')
st.subheader('Clustered Countries Data')
st.dataframe(df)

st.subheader('Summary Statistics')
st.write(df.describe())

st.subheader('Cluster Distribution')
cluster_counts = df['Cluster'].value_counts()
st.bar_chart(cluster_counts)
