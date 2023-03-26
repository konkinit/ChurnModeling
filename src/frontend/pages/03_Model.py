import os
import sys
import streamlit as st
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import Params_rdmf, evaluate_rdmf
Feature_Engineering = __import__(
                        "pages.02_Feature Engineering",
                        fromlist=["Modeling_Data"])
Modeling_Data = Feature_Engineering.Modeling_Data

st.markdown("# Modeling")

st.markdown("To sum up, the modling part starts with data resumed \
in the following table")

st.markdown("Since we deal with sparse matrix as inputs , mainly \
tree-based models are implmented and compared. ")

modeling_data = Modeling_Data()


st.markdown("## Random Forest")

rdmf_params = Params_rdmf(
    X_train=modeling_data.X_train_sparse,
    y_train=modeling_data.y_train,
    _n_estimators=50,
    _max_depth=2)

evaluate_rdmf(rdmf_params)


st.markdown("## XGBoost")
