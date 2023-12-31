import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
import seaborn as sns
from sklearn.metrics import silhouette_samples, silhouette_score

st.set_option('deprecation.showPyplotGlobalUse', False)

#Usia
def label_encoder():
    data['DIAGNOSA'] = LabelEncoder().fit_transform(data['DIAGNOSA'])
    data['JEN. KEL'] = LabelEncoder().fit_transform(data['JEN. KEL'])
    

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
    with st.expander("Data"):
        st.dataframe(df)

    with st.expander("Chart Usia Pasien"):
        plt.figure(figsize=(8, 6))
        sns.histplot(data=df, x='USIA', bins=20, kde=True)
        st.pyplot()

    st.markdown('---')
    
#Algoritma K-Means
    st.markdown("## K-Means")

    data = df[['NO','JEN. KEL','USIA','DIAGNOSA']]

    with st.expander("Data Fitur"):
        st.dataframe(data)

    label_encoder()

    kol_cluster = st.multiselect(
    "Pilih Kolom Untuk Clustering",
    ['JEN. KEL','USIA','DIAGNOSA'],
    default=['DIAGNOSA','USIA'],
    )
    X = data[kol_cluster+['NO']]

    penjelasan_k = ''' K pada K-means clustering menandakan jumlah kluster yang digunakan. '''
    nilai_k = st.slider("Pilih Nilai 'K'", min_value=2, max_value=10, value=5, help=penjelasan_k)
    
    kmeans = KMeans(nilai_k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(data)

    with st.expander("Data Fitur"):
        st.dataframe(data)

# Sidebar elbow
    with st.expander("Elbow Method"):
        distorsi = []
        k_tes = range(2, 10)
        for k in k_tes:
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
            kmeans.fit(data[['DIAGNOSA', 'USIA']])
            distorsi.append(kmeans.inertia_)

        plt.figure(figsize=(8, 6))
        plt.plot(k_tes, distorsi, 'bx-')
        plt.xlabel('Number of Clusters (k)')
        plt.ylabel('Distortion')
        plt.title('Elbow Method')
        st.pyplot()

# Scatter Plot 2D
    pilih_x = st.selectbox('Pilih Kolom x:', ('DIAGNOSA','JEN. KEL','USIA'))
    pilih_y = st.selectbox('Pilih Kolom y:', ('USIA','DIAGNOSA','JEN. KEL'))
    st.write(pilih_x)
    fig, ax = plt.subplots()
    plt.style.context('seaborn-whitegrid')
    ax.scatter(data[pilih_x], data[pilih_y], c=labels)
    ax.set_xlabel(pilih_x)
    ax.set_ylabel(pilih_y)
    st.pyplot(fig)

# Scatter Plot 3D
    fig = px.scatter_3d(data, x='DIAGNOSA', y='USIA', z='JEN. KEL',color=labels)
    st.plotly_chart(fig, use_container_width=True)