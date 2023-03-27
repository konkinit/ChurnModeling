import os
import sys
from pickle import load
import streamlit as st
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import Params_rdmf, evaluate_rdmf

st.markdown("# Modeling")

st.markdown("To sum up, the modling part starts with data resumed \
in the following table")

st.markdown("Since we deal with sparse matrix as inputs , mainly \
tree-based models are implmented and compared. ")


Modeling_Data = load(open('./data/app_inputs/modeling_data.pkl', 'rb'))

st.markdown("## Random Forest")

rdmf_params = Params_rdmf(
    X_train=Modeling_Data.X_train_sparse,
    y_train=Modeling_Data.y_train,
    _n_estimators=50,
    _max_depth=2)

st.markdown(f"On training data , the model achieves an accuracy \
of {round(100*evaluate_rdmf(rdmf_params), 3)} %")


st.markdown("## XGBoost")
