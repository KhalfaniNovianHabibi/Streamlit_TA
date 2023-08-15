import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
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
    with st.expander("Data"):
        st.dataframe(df)
    with st.expander("Data"):
        st.dataframe(df[['NO','JEN. KEL','USIA','DIAGNOSA']])

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

    kmeans = KMeans(nilai_k, random_state=0, n_init=10)
    kmeans.fit(df[['diagnosa_encoded','usia_encoded']])

    fig, ax = plt.subplots(figsize=(16, 9))
    #Create scatterplot
    ax = sns.scatterplot(
        ax=ax,
        x=df.diagnosa_encoded,
        y=df.usia_encoded,
        hue=kmeans.labels_,
        palette=sns.color_palette("colorblind", n_colors=nilai_k),
        legend=None,
    )
    fig