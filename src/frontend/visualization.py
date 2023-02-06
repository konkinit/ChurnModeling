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

# data
data = import_from_local(".")

## metadata variable management pipeline
data.iloc[:5, :]

### splitting

st.markdown('The dataset is described by 112 features and 56 124 rows. To train the model splittingg the data is required to measure performance.')
split_frac = st.text_input('Fraction of validation data')