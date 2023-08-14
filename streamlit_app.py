
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