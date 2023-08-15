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





#Usia
def label_encoder():
    df['DIAGNOSA'] = LabelEncoder().fit_transform(df['DIAGNOSA'])
    df['JEN. KEL'] = LabelEncoder().fit_transform(df['JEN. KEL'])
    

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

    st.markdown('---')
    
    #Algoritma K-Means
    st.markdown("## K-Means")

    with st.expander("Data Fitur"):
        st.dataframe(df[['NO','JEN. KEL','USIA','DIAGNOSA']])

    label_encoder()
    kol_cluster = st.multiselect(
    "Pilih Kolom Untuk Clustering",
    ['JEN. KEL','USIA','DIAGNOSA'],
    default=['DIAGNOSA','USIA'],
    )
    X = df[kol_cluster+['NO']]

    penjelasan_k = ''' K pada K-means clustering menandakan jumlah kluster yang digunakan. '''
    nilai_k = st.slider("Pilih Nilai 'K'", min_value=2, max_value=10, value=5, help=penjelasan_k)
    
    kmeans = KMeans(nilai_k, random_state=0, n_init=10)
    labels = kmeans.fit_predict(X)

    # Visualization
    pilih_x = st.selectbox('Pilih Kolom x:', ('JEN. KEL','USIA','DIAGNOSA'))
    pilih_y = st.selectbox('Pilih Kolom y:', ('JEN. KEL','USIA','DIAGNOSA'))
    st.write(pilih_x)
    plt.style.context('seaborn-whitegrid')
    plt.scatter(df[[pilih_x]].collect()[pilih_x],
                df[[pilih_y]].collect()[pilih_y],
                c=labels[['CLUSTER_ID']].collect()['CLUSTER_ID'])
    plt.xlabel(pilih_x)
    plt.ylabel(pilih_y)
    st.pyplot()