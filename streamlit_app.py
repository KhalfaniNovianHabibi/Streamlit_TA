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

# Scatter Plot 2D
    pilih_x = st.selectbox('Pilih Kolom x:', ('DIAGNOSA','JEN. KEL','USIA'))
    pilih_y = st.selectbox('Pilih Kolom y:', ('USIA','DIAGNOSA','JEN. KEL'))
    st.write(pilih_x)
    plt.style.context('seaborn-whitegrid')
    plt.scatter(data[pilih_x], data[pilih_y])
    plt.xlabel(pilih_x)
    plt.ylabel(pilih_y)
    st.pyplot()

# Scatter Plot 3D
    fig = px.scatter_3d(data, x='DIAGNOSA', y='USIA', z='JEN. KEL')
    st.plotly_chart(fig, use_container_width=True)

# Silhouette Score
    avg_silh_by_cluster = labels.agg([('avg','SLIGHT_SILHOUETTE','SLIGHT_SILHOUETTE')],\
                                           group_by='CLUSTER_ID')
    silhouette_avg = labels.agg([('avg','SLIGHT_SILHOUETTE','SLIGHT_SILHOUETTE')]).values[0][0]
    n_clusters=len(avg_silh_by_cluster)
    y_lower = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = labels.filter(f'CLUSTER_ID={i}')[['SLIGHT_SILHOUETTE']]\
                                        .collect()['SLIGHT_SILHOUETTE'].values
 
        ith_cluster_silhouette_values.sort()
 
        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        plt.title(f'AVG silhouette - {silhouette_avg}')
        plt.vlines(silhouette_avg,y_lower,y_upper,color='red',linestyles='--')
        color = cm.nipy_spectral(float(i) / n_clusters)
        plt.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)
        y_lower = y_upper + 10
    st.sidebar.pyplot()
    st.write('Silhouette avg score = ',silhouette_avg)