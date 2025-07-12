import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Upload Dataset", "Data Overview", "Clustering", "Visualization"])

st.title("🛍️ Customer Segmentation using K-Means Clustering")

# --- Global Variable ---
agg_df = None

# --- Upload Dataset ---
if page == "Upload Dataset":
    st.header("Upload Customer Dataset")
    uploaded_file = st.file_uploader("📂 Upload Excel file (Online Retail Dataset)", type=["xlsx"])

    if uploaded_file is not None:
        try:
            df = pd.read_excel(uploaded_file, sheet_name=0)
            df.dropna(subset=["Customer ID"], inplace=True)
            df = df[df["Quantity"] > 0]
            df = df[df["Price"] > 0.0]
            df["SalesLineTotal"] = df["Quantity"] * df["Price"]

            agg_df = df.groupby(by="Customer ID", as_index=False).agg(
                MonetaryValue=("SalesLineTotal", "sum"),
                Frequency=("Invoice", "nunique"),
                LastInvoiceDate=("InvoiceDate", "max")
            )

            max_invoice_date = agg_df["LastInvoiceDate"].max()
            agg_df["Recency"] = (max_invoice_date - agg_df["LastInvoiceDate"]).dt.days

            st.session_state['agg_df'] = agg_df
            st.success("✅ File uploaded and processed successfully.")

        except Exception as e:
            st.error(f"❌ Failed to load Excel sheet: {e}")
    else:
        st.info("Upload a file to get started.")

# --- Data Overview ---
elif page == "Data Overview":
    if 'agg_df' not in st.session_state:
        st.warning("⚠️ Please upload and process the data first from Upload Dataset page.")
    else:
        agg_df = st.session_state['agg_df']
        st.header("📊 Aggregated RFM Data")
        st.dataframe(agg_df.head())

        st.subheader("📈 Distribution Plots")
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].hist(agg_df['MonetaryValue'], bins=10, color='skyblue', edgecolor='black')
        axs[0].set_title('Monetary Value')

        axs[1].hist(agg_df['Frequency'], bins=10, color='lightgreen', edgecolor='black')
        axs[1].set_title('Frequency')

        axs[2].hist(agg_df['Recency'], bins=10, color='salmon', edgecolor='black')
        axs[2].set_title('Recency')

        st.pyplot(fig)

# --- Clustering Page ---
elif page == "Clustering":
    if 'agg_df' not in st.session_state:
        st.warning("⚠️ Please upload and process the data first from Upload Dataset page.")
    else:
        agg_df = st.session_state['agg_df']
        non_outliers_df = agg_df.copy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])

        st.subheader("📌 Elbow Method Graph")
        wcss = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
            kmeans.fit(scaled_data)
            wcss.append(kmeans.inertia_)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(range(1, 11), wcss, marker='o', linestyle='--')
        ax.set_title("The Elbow Point Graph")
        ax.set_xlabel("Number of clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

# --- Visualization Page ---
elif page == "Visualization":
    if 'agg_df' not in st.session_state:
        st.warning("⚠️ Please upload and process the data first from Upload Dataset page.")
    else:
        agg_df = st.session_state['agg_df']
        non_outliers_df = agg_df.copy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])

        k = st.slider("Select number of clusters for visualization", min_value=2, max_value=10, value=4)
        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        non_outliers_df["Cluster"] = cluster_labels

        # Inverse transform centroids for interpretation
        centroids = scaler.inverse_transform(kmeans.cluster_centers_)
        centroids_df = pd.DataFrame(centroids, columns=["MonetaryValue", "Frequency", "Recency"])

        st.subheader("📋 Cluster Counts")
        st.bar_chart(non_outliers_df['Cluster'].value_counts())

        st.subheader("📌 Cluster Averages")
        st.dataframe(non_outliers_df.groupby('Cluster')[["MonetaryValue", "Frequency", "Recency"]].mean())

        st.subheader("📍 Cluster Centroids")
        st.dataframe(centroids_df)

        st.subheader("💾 Download Clustered Data")
        csv = non_outliers_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download as CSV", data=csv, file_name='clustered_customers.csv', mime='text/csv')
