import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import silhouette_samples, silhouette_score

#Usia
def modifikasi_usia():
    df['USIA'] = pd.cut(
        x=df['USIA'],
        bins=[0,17,64,np.inf],
        labels=['Anak-Anak','Dewasa','Lansia']
        )

# Streamlit
with st.sidebar:
    st.sidebar.title("Source Code")
    st.sidebar.info(
        """
        <https://github.com/KhalfaniNovianHabibi/Streamlit_TA/>
        """
    )
    uploaded_file = st.file_uploader("Unggah File XLSX", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file, skiprows=[0,1])
    st.title("Aplikasi Streamlit")
    st.markdown("## Dataset")
    st.write('Jumlah Data :',len(df))
    with st.expander("Data Excel"):
        modifikasi_usia()
        st.dataframe(df)
    
    st.markdown('---')
    
    #Algoritma K-Means
    st.markdown("## K-Means")

    penjelasan_k = ''' K pada K-means clustering menandakan jumlah kluster yang digunakan. '''
    nilai_k = st.slider("Pilih Nilai 'K'", min_value=1,
                        max_value=15, value=5,
                        help=penjelasan_k)
    
    label_encoder = LabelEncoder()
    df['diagnosa_encoded'] = label_encoder.fit_transform(df['DIAGNOSA'])
    df['usia_encoded'] = label_encoder.fit_transform(df['USIA'])

    kmeans = KMeans(n_clusters=nilai_k, random_state=0, n_init=10)
    kmeans.fit(df[['diagnosa_encoded','usia_encoded']])

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_

    # Plot the data points and cluster centroids
    plt.figure(figsize=(8, 6))
    plt.scatter(df[:, 0], df[:, 1], c=labels, cmap='rainbow', marker='o', edgecolors='k', alpha=0.7)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='black', marker='X', s=100, label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'K-Means Clustering with {nilai_k} Clusters')
    plt.legend()
    st.pyplot(plt)

    # Display cluster centroids
    st.subheader("Cluster Centroids")
    st.write(centroids)

    # Display cluster assignments for each data point
    st.subheader("Cluster Assignments")
    st.write(labels)