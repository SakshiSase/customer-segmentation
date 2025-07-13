import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import BytesIO

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# Navigation menu
menu = st.sidebar.selectbox("Navigation", ["🏠 Home", "📊 Data Overview", "🧠 Clustering", "📁 Results"])

# Global variables
if "df" not in st.session_state:
    st.session_state.df = None
if "clustered_data" not in st.session_state:
    st.session_state.clustered_data = None

# Home Page
if menu == "🏠 Home":
    st.title("🧩 Customer Segmentation ML App")
    st.markdown("""
    Welcome to the Customer Segmentation App using Machine Learning (KMeans Clustering).
    
    - Upload your customer data as a `.csv` file.
    - Select features for clustering.
    - View visualizations and download the result.
    """)
    uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("✅ Dataset uploaded successfully!")

# Data Overview
elif menu == "📊 Data Overview":
    st.title("📊 Dataset Overview")
    if st.session_state.df is not None:
        df = st.session_state.df
        st.write("### First 5 Rows of the Dataset")
        st.write(df.head())
        st.write("### Dataset Info")
        st.write(df.describe())
        
        if st.checkbox("Show Heatmap"):
            plt.figure(figsize=(10, 6))
            sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
            st.pyplot(plt.gcf())
        
        if st.checkbox("Show Pairplot (slow for large datasets)"):
            st.info("Plotting pairplot... please wait.")
            fig = sns.pairplot(df)
            st.pyplot(fig)
    else:
        st.warning("Please upload the dataset from the Home page.")

# Clustering
elif menu == "🧠 Clustering":
    st.title("🧠 KMeans Clustering")
    if st.session_state.df is not None:
        df = st.session_state.df

        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        selected_features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:2])

        if selected_features:
            k = st.slider("Choose number of clusters", 2, 10, 3)
            X = df[selected_features]
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            model = KMeans(n_clusters=k, random_state=42)
            df['Cluster'] = model.fit_predict(X_scaled)
            st.session_state.clustered_data = df

            st.success(f"✅ Clustering complete with {k} clusters!")
            st.write(df.head())

            # Visualization
            if len(selected_features) >= 2:
                fig, ax = plt.subplots()
                sns.scatterplot(x=df[selected_features[0]], y=df[selected_features[1]], hue=df['Cluster'], palette='Set2', ax=ax)
                ax.set_title("Customer Segmentation Scatterplot")
                st.pyplot(fig)
        else:
            st.info("Please select at least two features.")
    else:
        st.warning("Upload dataset first on Home page.")

# Results
elif menu == "📁 Results":
    st.title("📁 Final Results & Download")
    if st.session_state.clustered_data is not None:
        st.write("### Clustered Dataset Preview")
        st.write(st.session_state.clustered_data.head())

        csv = st.session_state.clustered_data.to_csv(index=False)
        b64 = BytesIO()
        b64.write(csv.encode())
        b64.seek(0)

        st.download_button("📥 Download Clustered CSV", b64, "clustered_result.csv", "text/csv")
    else:
        st.warning("No clustered data available. Perform clustering first.")
