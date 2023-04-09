import os
import sys
import streamlit as st
from pandas import DataFrame
from pickle import load
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.models import (
    train_rdmf,
    evaluate_rdmf
)
from src.utils import Params_rdmf


Modeling_Data = load(
    open('./data/app_inputs/modeling_data.pkl', 'rb'))


st.markdown("# Modeling")

st.markdown("To sum up, the modling part starts with data resumed \
in the following table")

st.dataframe(
    data=DataFrame(
        columns=[""]
    ),
    use_container_width=True
)

st.markdown("Since we deal with sparse matrix as inputs , mainly \
tree-based models are implmented and compared. ")


st.markdown("## Random Forest")
st.markdown("It is a classification task then predicting a class for an obs. \
depends on a certain cutoff which have a default value of 0.5. \
Model performance is going to be monitored with some ranged values of cutoffs")
rdmf_params = Params_rdmf(
    X_train=Modeling_Data.X_train_sparse,
    y_train=Modeling_Data.y_train,
    X_valid=Modeling_Data.X_valid_sparse,
    y_valid=Modeling_Data.y_valid,
    _n_estimators=100,
    _max_depth=2)
train_rdmf(rdmf_params)


_cutoff = st.slider(
                label='Choose the cutoff',
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05)


(_report_train,
 _acc_train,
 _cm_train,
 _report_valid,
 _acc_valid,
 _cm_valid) = evaluate_rdmf(rdmf_params, _cutoff)

st.markdown(f"The accuracies on the training and valid datasets are \
{round(100*_acc_train, 2)} % and {round(100*_acc_valid, 2)} % respectively \
according to a cutoff of {_cutoff}")

st.markdown("The classification report on both training and validation \
datasets are listed below")
st.dataframe(
    data=_cm_train,
    use_container_width=False)
st.dataframe(
    data=_cm_valid,
    use_container_width=False)
st.dataframe(
    data=_report_train,
    use_container_width=True)
st.dataframe(
    data=_report_valid,
    use_container_width=True)


st.markdown("## XGBoost")
