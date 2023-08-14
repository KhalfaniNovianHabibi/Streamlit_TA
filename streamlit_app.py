import streamlit as st
import pandas as pd
import numpy as np
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
    with st.expander("Data Asli"):
        modifikasi_usia()
        st.dataframe(df)
    
    st.markdown('---')