import os
import sys
import streamlit as st
from plotly.express import pie

if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.data import import_data


st.markdown("# Target variable: `churn`")
st.markdown("## Description")

st.markdown(
    """
    `churn` is a binary variable distributed as below without
    missingness 87.9 % of the customers are not churned in contrast to 12.1
    % who churned.
    """
)

raw_data = import_data()

fig = pie(raw_data, "churn", width=450, height=450)

st.plotly_chart(fig, use_container_width=True)

st.markdown("### Fraction of splitting to train and valid data")
st.markdown(
    """
    Before feature engineering starts, it is suitable to perform
    data splitting in order to avoid information leaks. Here a stratify
    splitting is performed.
    """
)

train_frac = st.slider(
    label='Choose the percent of the training data set',
    min_value=50,
    max_value=100,
    value=70,
    step=5
)

st.session_state["train_frac"] = train_frac/100
