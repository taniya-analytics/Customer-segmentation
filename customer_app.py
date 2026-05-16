import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set Streamlit page config
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# Load data
data = pd.read_csv('Customer_Segments.csv')

# Identify PCA columns
pca_cols = ['PCA1', 'PCA2'] if 'PCA1' in data.columns else ['PC1', 'PC2']

# ---------------- Session State Handling ---------------- #
if 'page' not in st.session_state:
    st.session_state.page = 'welcome'

def go_to_dashboard():
    st.session_state.page = 'dashboard'

# ---------------- Welcome Page ---------------- #
if st.session_state.page == 'welcome':
    st.markdown("""
        <div style='text-align: center;'>
            <h1>🧑‍💼 Customer Segmentation using Machine Learning</h1>
            <h3>Final Project - Credit Card Customer Segmentation</h3>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ---
    ### 📋 Project Overview

    This project performs *Customer Segmentation* using *Principal Component Analysis (PCA)* and *KMeans Clustering* on a credit card dataset.

    The goal is to identify distinct customer groups based on:
    - 💳 Transaction behavior
    - 📈 Credit utilization
    - 📊 Engagement patterns
    - 💼 Financial attributes

    Such segmentation helps businesses improve targeting, reduce churn, and personalize customer experience.
    """)

    st.markdown("### 👇 Click the button below to explore the dashboard:")
    st.button("🚀 Enter Dashboard", on_click=go_to_dashboard)

# ---------------- Dashboard Page ---------------- #
elif st.session_state.page == 'dashboard':
    # Sidebar
    st.sidebar.title("Explore Segments")
    clusters = sorted(data['Cluster'].unique())
    selected_cluster = st.sidebar.selectbox("Select Cluster", clusters)

    # Silhouette Score (if exists)
    if 'Silhouette' in data.columns:
        st.sidebar.markdown(f"*Silhouette Score*: {round(data['Silhouette'].iloc[0], 4)}")

    # Title
    st.title("💳 Customer Segmentation Dashboard")

    # Cluster Stats
    st.subheader(f"📊 Statistics for Cluster {selected_cluster}")
    st.write(data[data['Cluster'] == selected_cluster].describe())

    # PCA Plot
    st.subheader("🌀 PCA Cluster Visualization")
    fig, ax = plt.subplots()
    for cluster in clusters:
        cluster_data = data[data['Cluster'] == cluster]
        ax.scatter(cluster_data[pca_cols[0]], cluster_data[pca_cols[1]], label=f'Cluster {cluster}', s=50)
    ax.set_xlabel(pca_cols[0])
    ax.set_ylabel(pca_cols[1])
    ax.set_title("Customer Segments Visualized")
    ax.legend()
    st.pyplot(fig)

    # Feature Distribution
    st.subheader("📈 Feature Distribution in Selected Cluster")
    valid_features = data.columns.drop(['Cluster'] + pca_cols)
    selected_features = st.multiselect("Select Features to Explore", valid_features)

    if selected_features:
        for feature in selected_features:
            fig, ax = plt.subplots()
            sns.histplot(data=data[data['Cluster'] == selected_cluster], x=feature, kde=True, ax=ax)
            ax.set_title(f"{feature} Distribution in Cluster {selected_cluster}")
            st.pyplot(fig)