import sys
sys.path.append("/home/ikonkobo/Desktop/Self_Learning/telco_churn/")
from pandas import DataFrame
from numpy import log
import streamlit as st
import plotly.express as px
from src.data.metadata_analysis import MetadataStats
from src.visualization.plotly_func import df_skewed_feature
from src.data.import_data import import_from_local
from src.data.train_valid_split import train_valid_split
from src.features.build_features import MetaDataManagement, DataManagement
from src.models.random_forest import evaluate_rdmf

st.set_page_config(page_title="Data Processing & Feature Engineering")

raw_data = import_from_local(".")

st.markdown("Since ML model perform well with no leak of informations \
between train and test data sets, it comes to distinguish two levels of data engineering")

st.markdown("## Metadata Level")

st.markdown("Below is the metadata tables, one for numerocal data type and the other \
for character data type. They describe the raw properties of each feature. \
for better understanding and better decision")

st.dataframe(
    data=MetadataStats(raw_data).metadata_report('char'),
    use_container_width=True)
st.dataframe(
    data=MetadataStats(raw_data).metadata_report('num'),
    use_container_width=True)

st.markdown("### Numerical features processing")
st.markdown("A starting point of this section is the identification of \
useless features and their removing. Afterwards, due to some business \
assumptions some features cannot have a certain values like negative \
one or must be transformed in order to handle skewness or kurtosis.")


st.markdown("\
* The following features can be handled as useless in view of the use case `issue_level2`,\
`resolution`, `city`, `city_lat`, `city_long`, `data_usage_amt`, \
`mou_onnet_6m_normal`, `mou_roam_6m_normal`, `region_lat`, `region_long`, `state_lat`, \
`state_long`, `tweedie_adjusted`, `upsell_xsell`; \n \
* Some varoable especially those having `MB_Data_Usg_M0` are very skewed as the plot describes. \
A $\log$ transformation here is suitable to handle log_MB_Data_Usg_M0_i = $\log$ ( 1 + MB_Data_Usg_M0_i )")

feature = st.selectbox(
    "Choose a variable to see its distribution",
    (f"MB_Data_Usg_M0{str(i)}" for i in range(4, 10)))

log_feature = st.radio(
    "Choose the values to be plotted: raw or log-transformed",
    (feature, f"log_{feature}"))

fig = px.histogram(
            df_skewed_feature(raw_data, feature), 
            x=log_feature, 
            nbins=30, 
            histnorm='probability density'
            )
st.plotly_chart(fig, use_container_width=True)

st.markdown("* It is not reasonnable for some variables such as to have negative values. \
To handle this incoherence the $ReLU$ is applied to censor the value")


st.markdown("### Categorical & Text features processing")

st.markdown("In order to outcome to same data structure for the both train and valid data sets, the decisions here are:\n \
* For categorical features the method employed is OneHotEncoding \
with missing values, if they exist, form a category.\n \
* The text variable which is `verbatims` will be handled after splitting the data set into train and \
valid partition in order to avoid information leaking. Indeed the idea here is to build a text mining model \
on the train data and afterwards score the valid data set ")


st.markdown("## Data Level")

