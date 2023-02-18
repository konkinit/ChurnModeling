import sys
sys.path.append("/home/ikonkobo/Desktop/Self_Learning/telco_churn/")
from pandas import DataFrame
from numpy import log, random
import streamlit as st
import plotly.express as px
from src.data.import_data import import_from_local
from src.data.checkers import train_frac_check
from src.data.save_data import *

st.markdown("# Target variable `churn`")

st.markdown("### Description")
st.markdown("`churn` is a binary variable distributed as below without missingness \
87.9 % of the customers are not churned in contrast to 12.1 % who churned.  ")

raw_data = import_from_local(".")


fig = px.pie(
            raw_data, "churn",
            width=400, height=400
            )
st.plotly_chart(fig, use_container_width=True)

st.markdown("### Fraction of splitting to train and valid data")
st.markdown("Before feature engineering starts, it is suitable to perform data splitting \
in order to avoid information leaks. Here a startify splitting is performed.")

train_frac = st.slider(label='Choose the percent of the training data set', 
                        min_value=50, 
                        max_value=100, 
                        value=70,
                        step=5
                    )

save_input_data("train_frac", train_frac/100)
    
