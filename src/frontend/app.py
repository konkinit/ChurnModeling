import streamlit as st

st.set_page_config(page_title="Onboarding")

st.title('ML App for Churn Prediction')

st.header('Description')
st.markdown('Churn prediction is one of greatest challenge faced by businesses.\
According to statistics retain a customer is five time cheaper than engage a new one. \

Nowadays machine learning methods offer huge capabilities to overcome this issue hence this project. \
Using a dataset obtained from SAS Machine Learning Specialist Course \
the aim is to build some ML models that predict the churn, compare them to choose the best one.')

st.header('Description of the data set')
st.markdown("The data are from the Telco industry and are described by 56 557 customers' \
behavioural, demographical attributes characterized by 110 features. Besides the dataset \
has already a churn label and as cited in the below table there are churners against non churners")
