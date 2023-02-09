import os
import streamlit as st
import sys
sys.path.append("/home/ikonkobo/Desktop/Self_Learning/telco_churn/")

from src.data.import_data import import_from_local
from src.data.train_valid_split import train_valid_split
from src.features.build_features import MetaDataManagement, DataManagement
from src.models.random_forest import evaluate_rdmf

st.title('Churn prediction in Telco industry')

st.header('Description')
st.markdown('The project aims to predict the churn using data from Telco industry using some machine learning algorithms')

st.header('Data Processing & Feature Engineering')
st.markdown('The dataset is described by 112 features and 56 557 rows. To train the model splittingg the data is required to measure performance.')

# data
data = import_from_local(".")
MetaDataManagement(data).metadata_management_pipeline()
data.head()

"""
### splitting
X_train, X_valid, y_train, y_valid = train_valid_split(data)
    
## data variable management pipeline
DataManagement(X_train).data_management_pipeline()
DataManagement(X_valid).data_management_pipeline()


# models pipelines
## random forest model
evaluate_rdmf(X_train, X_valid, y_train, y_valid, n_estimators=20)

# model comparaison
"""