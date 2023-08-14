import streamlit as st
import pandas as pd
import numpy as np

# Streamlit
with st.sidebar:
    st.sidebar.title("Source Code")
    st.sidebar.info(
        """
        <https://github.com/TetukoAnglingKusumo/STREAMLIT-APP>
        """
    )
    uploaded_file = st.file_uploader("Upload File XLSX", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file, skiprows=[0,1])
    st.title("Streamlit Apps")
    st.markdown("## Dataset")
    st.write('Jumlah Data :',len(df))
    with st.expander("Expand **Raw Data**"):
        mod_data()
        st.dataframe(df)