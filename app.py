import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

st.set_page_config(page_title="Customer Segmentation App", layout="wide")

# --- Sidebar Navigation ---
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Explore", "Clustering & Results", "About"])

st.title("🛍️ Customer Segmentation using K-Means Clustering")

# --- Home Page ---
if page == "Home":
    st.header("Welcome 👋")
    st.markdown("""
    This Streamlit app performs **customer segmentation** using **K-Means Clustering** based on Recency, Frequency, and Monetary (RFM) values.

    **How it works:**
    - Upload your Online Retail Excel data
    - Explore the RFM metrics
    - Run K-Means clustering
    - Visualize your customer groups in 3D and download the results

    > ⚠️ Please make sure the Excel file includes columns like `Invoice`, `Quantity`, `Price`, `Customer ID`, and `InvoiceDate`.
    """)

# --- Upload & Explore Page ---
elif page == "Upload & Explore":
    uploaded_file = st.file_uploader("📂 Upload Excel file (Online Retail Dataset)", type=["xlsx"])

    if uploaded_file is not None:
        sheet_name = st.text_input("Enter Sheet Name (default: 0)", value="0")
        try:
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

            st.subheader("📊 Aggregated RFM Data")
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

            st.session_state['agg_df'] = agg_df

        except Exception as e:
            st.error(f"❌ Failed to load Excel sheet: {e}")
    else:
        st.warning("Please upload a valid Excel (.xlsx) file.")

# --- Clustering Page ---
elif page == "Clustering & Results":
    if 'agg_df' not in st.session_state:
        st.warning("⚠️ Please upload and process the data from the 'Upload & Explore' section first.")
    else:
        agg_df = st.session_state['agg_df']
        non_outliers_df = agg_df.copy()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(non_outliers_df[["MonetaryValue", "Frequency", "Recency"]])

        st.subheader("📌 Choose number of clusters (k)")
        k = st.slider("Select k", min_value=2, max_value=10, value=4)

        kmeans = KMeans(n_clusters=k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        non_outliers_df["Cluster"] = cluster_labels

        st.subheader("🎯 3D Cluster Visualization")
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

        st.subheader("📌 Cluster Averages")
        st.dataframe(non_outliers_df.groupby('Cluster')[["MonetaryValue", "Frequency", "Recency"]].mean())

        st.subheader("💾 Download Clustered Data")
        csv = non_outliers_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download as CSV", data=csv, file_name='clustered_customers.csv', mime='text/csv')

# --- About Page ---
elif page == "About":
    st.header("About This App")
    st.markdown("""
    This project is built using:
    - 🐍 Python
    - 📊 Pandas, Matplotlib, Seaborn
    - 🤖 Scikit-learn (KMeans Clustering)
    - 📈 Streamlit (Web Interface)

    Developed for **Customer Segmentation** using the RFM (Recency, Frequency, Monetary) model.
    
    👉 Upload your Excel data and explore insights instantly!
    """)
