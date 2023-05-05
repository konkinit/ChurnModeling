import os
import sys
import streamlit as st
from pandas import DataFrame
from pickle import load
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.utils import save_input_data
from src.models import (
    RandomForest_
)


Modeling_Data = load(
    open('./data/app_inputs/modeling_data.pkl', 'rb'))

st.markdown("# Modeling")

st.markdown("To sum up, the modling part starts with data resumed \
in the following table")

st.dataframe(
    data=DataFrame(
        data=[[
                Modeling_Data.X_train_sparse.shape[0],
                Modeling_Data.X_train_sparse.shape[1]
            ],
              [
                Modeling_Data.X_test_sparse.shape[0],
                Modeling_Data.X_test_sparse.shape[1]
            ]],
        columns=["Number of observation", "Number of features"],
        index=[["Training data", "Test data"]]
    ),
    use_container_width=False
)

st.markdown("Since we deal with sparse matrix as inputs , mainly \
tree-based models are implmented and compared. ")

st.markdown("## Random Forest")

st.markdown("It is a classification task then predicting a class for an obs. \
depends on a certain cutoff which have a default value of 0.5. \
Model performance is going to be monitored with some ranged values of cutoffs")

rdmf_model = RandomForest_(
    model_name="rdmf",
    X_train=Modeling_Data.X_train_sparse,
    y_train=Modeling_Data.y_train,
    X_test=Modeling_Data.X_test_sparse,
    y_test=Modeling_Data.y_test
)

rdmf_model.fit_and_save()

_cutoff = st.slider(
                label='Choose the cutoff',
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.05
            )

save_input_data(
    "rdmf_cutoff",
    _cutoff
)

(
    report_train, accuracy_train, cm_train,
    report_test, accuracy_test, cm_test
) = rdmf_model.inference(_cutoff)

st.markdown("The classification reports are listed below and \
refer respectively to training and test datasets")

col1, col2, col3 = st.columns(3)
col1.metric("Accuracy", f"{round(100*accuracy_train, 2)}")
col2.metric("Recall `active`", f"\
    {round(100*report_train['active']['recall'], 2)}")
col3.metric("Recall `churned`", f"\
    {round(100*report_train['churned']['recall'], 2)}")

st.dataframe(data=cm_train)

col1_, col2_, col3_ = st.columns(3)
col1_.metric("Accuracy", f"{round(100*accuracy_test, 2)}")
col2_.metric("Recall `active`", f"\
    {round(100*report_test['active']['recall'], 2)}")
col3_.metric("Recall `churned`", f"\
    {round(100*report_test['churned']['recall'], 2)}")
st.dataframe(data=cm_test)


st.markdown("## XGBoost")
