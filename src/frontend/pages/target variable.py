import sys
sys.path.append("/home/ikonkobo/Desktop/Self_Learning/telco_churn/")
from pandas import DataFrame
from numpy import log
import streamlit as st
import plotly.express as px
from src.data.import_data import import_from_local

st.markdown("# Target variable `churn`")

st.markdown("")

raw_data = import_from_local(".")