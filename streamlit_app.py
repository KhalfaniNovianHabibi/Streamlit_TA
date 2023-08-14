import streamlit as st
import pandas as pd
import numpy as np
from apyori import apriori

# Function
def mod_data():
    # Umur
    df['UMUR'] = pd.cut(
        x=df['UMUR'],
        bins=[0,5,11,25,45,65,np.inf],
        labels=['BALITA','KANAK-KANAK','REMAJA','DEWASA','LANSIA','MANULA']
        )

    # Alamat
    alamat  =   {'KELURAHAN BARU':'ARUT SELATAN',
                'PANGKALAN BUN':'ARUT SELATAN',
                'KUMAI HILIR':'KUMAI',
                'PANGKALAN BANTENG':'PANGKALAN BANTENG',
                'SUNGAI BENGKUANG':'PANGKALAN BANTENG',
                'MARGA MULYA':'PANGKALAN BANTENG',
                'SIMPANG BERAMBAI':'PANGKALAN BANTENG',
                'KARANG MULYA':'PANGKALAN BANTENG',
                'NATAI KERBAU':'PANGKALAN BANTENG',
                'SUNGAI PULAU':'PANGKALAN BANTENG',
                'SUNGAI HIJAU':'PANGKALAN BANTENG',
                'AMIN JAYA':'PANGKALAN BANTENG',
                'ARGA MULYA':'PANGKALAN BANTENG',
                'KEBUN AGUNG':'PANGKALAN BANTENG',
                'SUNGAI PAKIT':'PANGKALAN BANTENG',
                'BERAMBAI MAKMUR':'PANGKALAN BANTENG',
                'SUNGAI KUNING':'PANGKALAN BANTENG',
                'SIDO MULYO':'PANGKALAN BANTENG',
                'KARANG SARI':'PANGKALAN BANTENG',
                'MULYA JADI':'PANGKALAN BANTENG',
                'SEMANGGANG':'PANGKALAN BANTENG',
                'INTI 4':'PANGKALAN BANTENG',
                'INTI 1':'PANGKALAN BANTENG',
                'SEBUKAT':'PANGKALAN BANTENG',
                'PURBASARI':'PANGKALAN LADA',
                'PANGKALAN LADA':'PANGKALAN LADA',
                'PANDU SANJAYA':'PANGKALAN LADA',
                'PANGKALAN DEWA':'PANGKALAN LADA',
                'SIMPANG LADA':'PANGKALAN LADA',
                'PANGKALAN TIGA':'PANGKALAN LADA',
                'LADA MANDALA JAYA':'PANGKALAN LADA',
                'SUNGAI MELAWEN':'PANGKALAN LADA',
                'MAKARTI JAYA':'PANGKALAN LADA',
                'SUNGAI RANGIT':'PANGKALAN LADA',
                'SUMBER AGUNG':'PANGKALAN LADA',
                'KADIPI ATAS':'PANGKALAN LADA',
                'SIMPANG TIGA LADA':'PANGKALAN LADA',
                'PANGKUT':'ARUT UTARA',
                'PENYOMBAAN':'ARUT UTARA',
                'PANAHAN':'ARUT UTARA',
                'RIAM':'ARUT UTARA',
                'SUKAMANDANG':'LUAR KABUPATEN',
                'SUKA MAJU':'LUAR KABUPATEN',
                'PEMBUANG HULU':'LUAR KABUPATEN',
                'JEMPONG BARU':'LUAR KABUPATEN',
                'KELEBUH':'LUAR KABUPATEN',
                'LANGKOT':'LUAR KABUPATEN',
                'DASAN GERSIK':'LUAR KABUPATEN',
                'DANTI':'LUAR KABUPATEN',
                'BAKTI JAYA':'LUAR KABUPATEN',
                'BAUS':'LUAR KABUPATEN',
                'ASAM BARU':'LUAR KABUPATEN',
                'POLOS':'LUAR KABUPATEN',
                'SIMPANG SATU POLOS':'LUAR KABUPATEN',
                'RANTAU PULUT':'LUAR KABUPATEN'
            }
    df['ALAMAT']=df['ALAMAT'].map(alamat)

    # Penyakit
    penyakit = {'T1' :'DARAH',      'T2':'WIDAL',           'T3':'GLUKOSA',     'T4':'GLUKOSA',     'T5':'DARAH',
                'T6' :'DBD',        'T7':'DBD',             'T8':'KEHAMILAN',   'T9':'HIV',         'T10':'TRYGLICERIDE',
                'T11':'CHOLESTROL', 'T12':'HDL',            'T13':'LDL',        'T14':'UREUM',      'T15':'KREATININ',
                'T16':'ASAM URAT',  'T17':'ASAM URAT',      'T18':'PROTEIN',    'T19':'ALBUMIN',    'T20':'BILIRUBIN',
                'T21':'BILIRUBIN',  'T22':'SGOT',           'T23':'SGPT',       'T24':'BTA',        'T25':'URINE',
                'T26':'SEDIMEN',    'T27':'REDUKSI',        'T28':'PROTEIN',    'T29':'HEMOGLOBIN', 'T30':'LED',
                'T31':'CTBT',       'T32':'KUSTA',          'T33':'MALARIA',    'T34':'MALARIA',    'T35':'SHIFILIS',
                'T36':'HEPATITIS',  'T37':'MIKROFILARIA',   'T38':'GONOROE',    'T39':'JAMUR',      'T40':'BUN',
                'T44':'COVID-19',   'L':'LAIN-LAIN'
                }
    for i in range(1,11):
        df['P'+str(i)]=df['P'+str(i)].map(penyakit)

def preprocessing_data():
    dataset = df[['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']]

    dataset['DATA'] = dataset[dataset.columns[0:]].apply(
        lambda x: ','.join(x.dropna().astype(str)),
        axis=1
    )

    data = dataset[['DATA']]

    records = []
    for i in range(data.shape[0]):
        records.append([str(data.values[i,j]).split(',') for j in range(data.shape[1])])

    trx = [[] for trx in range(len(records))]
    for i in range(len(records)):
        for j in records[i][0]:
            trx[i].append(j)
    return trx


# Streamlit
with st.sidebar:
    st.sidebar.title("Batasan")
    st.sidebar.info(
        """
        Aplikasi ini dibuat khusus menyesuaikan dengan format data hasil pemeriksaan laboratorium Semanggang.
        
        Pastikan untuk menggunakan data yang sesuai dengan format tersebut agar hasilnya dapat ditampilkan dengan benar.
        
        Source code: <https://github.com/TetukoAnglingKusumo/STREAMLIT-APP>
        """
    )
    uploaded_file = st.file_uploader("Choose a XLSX file", type="xlsx")

if uploaded_file:
    df = pd.read_excel(uploaded_file, skiprows=[0,1])
    st.title("Streamlit Apps")
    st.markdown("## Dataset")
    st.write('Jumlah Data :',len(df))
    with st.expander("Expand **Raw Data**"):
        mod_data()
        st.dataframe(df)

    with st.expander("Expand **Bar Chart**"):
        disease_set = df[['P1','P2','P3','P4','P5','P6','P7','P8','P9','P10']]

        disease_set['DISEASE'] = disease_set[disease_set.columns[0:]].apply(
            lambda x: ','.join(x.dropna().astype(str)),
            axis=1
        )

        split_data = disease_set['DISEASE'].str.split(',').apply(pd.Series).stack().reset_index(level=1, drop=True).to_frame('DISEASE')

        disease_counts = split_data['DISEASE'].value_counts()

        st.bar_chart(disease_counts)

    st.markdown('---')
    st.markdown("## Apriori")
    st.markdown(
        '''
        **Support** menampilkan rekaman transaksi yang meliputi pembelian item yang terkait dalam satu kali proses transaksi.
            
        **Confidence** menampilkan catatan transaksi yang menggambarkan pembelian item dalam susunan berurutan, satu demi satu, dalam satu proses transaksi.
        '''
    )

    st.markdown('Support dan Confidence untuk himpunan item A dan B, dapat diwakili dengan rumus.')

    support_helper = ''' > Support(A) = (Jumlah transaksi dimana item A muncul) / (Jumlah total transaksi) '''
    confidence_helper = ''' > Confidence(A->B) = Support (Himpunan gabungan A dan B) / Support (A) '''
    st.markdown('---')

    support = st.slider("Masukkan Nilai Batas Minimum Support", min_value=0.1,
                        max_value=0.9, value=0.15,
                        help=support_helper)

    confidence = st.slider("Masukkan Nilai Batas Minimum Confidence", min_value=0.1,
                        max_value=0.9, value=0.6, help=confidence_helper)

    association_rules = apriori(preprocessing_data(), min_support=support, min_confidence=confidence,min_lift=1)
    association_results = association_rules

    pd.set_option('max_colwidth', 1000)

    Result=pd.DataFrame(columns=['Rule','Support','Confidence'])
    for item in association_results:
        pair = item[2]
        for i in pair:
            items = str([x for x in i[0]])
            if i[3]!=1:
                Result=Result.append({
                    'Rule':str([x for x in i[0]])+ " -> " +str([x for x in i[1]]),
                    'Support':str(round(item[1]*100,2))+'%',
                    'Confidence':str(round(i[2] *100,2))+'%'
                    },ignore_index=True)
    Result