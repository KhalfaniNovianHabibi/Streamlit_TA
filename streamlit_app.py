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
    uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")