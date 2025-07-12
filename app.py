import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Customer Segmentation App", layout="wide")
st.title("🛍️ Customer Segmentation using K-Means Clustering")

uploaded_file = st.file_uploader("Upload Excel file (Online Retail Dataset)", type=["xlsx"])

if uploaded_file:
    sheet_name = st.text_input("Enter Sheet Name (default: 0)", value="0")
    df = pd.read_excel(uploaded_file, sheet_name=int(sheet_name) if sheet_name.isdigit() else sheet_name)
    st.subheader("Sample Data")
    st.dataframe(df.head(10))

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

    st.subheader("Aggregated RFM Data")
    st.dataframe(agg_df.head())

    st.subheader("📊 Distribution Plots")
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    axs[0].hist(agg_df['MonetaryValue'], bins=10, color='skyblue', edgecolor='black')
    axs[0].set_title('Monetary Value')

    axs[1].hist(agg_df['Frequency'], bins=10, color='lightgreen', edgecolor='black')
    axs[1].set_title('Frequency')

    axs[2].hist(agg_df['Recency'], bins=10, color='salmon', edgecolor='black')
    axs[2].set_title('Recency')

    st.pyplot(fig)

    non_outliers_df = agg_df.copy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])

    st.subheader("📈 Choose number of clusters (k)")
    k = st.slider("Select k", min_value=2, max_value=10, value=4)

    kmeans = KMeans(n_clusters=k, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_data)
    non_outliers_df["Cluster"] = cluster_labels

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(non_outliers_df['MonetaryValue'],
                         non_outliers_df['Frequency'],
                         non_outliers_df['Recency'],
                         c=non_outliers_df['Cluster'], cmap='Set1')
    ax.set_xlabel("MonetaryValue")
    ax.set_ylabel("Frequency")
    ax.set_zlabel("Recency")
    st.pyplot(fig)

    st.subheader("📋 Cluster Counts")
    st.bar_chart(non_outliers_df['Cluster'].value_counts())

    st.subheader("📌 Cluster Means")
    st.dataframe(non_outliers_df.groupby('Cluster')[['MonetaryValue', 'Frequency', 'Recency']].mean())

    st.subheader("💾 Download Clustered Data")
    csv = non_outliers_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download as CSV", data=csv, file_name='clustered_customers.csv', mime='text/csv')

    st.success("✅ Customer Segmentation process completed successfully.")
else:
    st.info("Please upload a valid Excel file with the Online Retail dataset.")
