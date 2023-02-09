"""Main script of the project."""
import os
from src.data.import_data import import_from_local
from src.data.train_valid_split import train_valid_split
from src.features.build_features import MetaDataManagement, DataManagement
from src.models.random_forest import evaluate_rdmf

os.chdir("/home/ikonkobo/Desktop/Self_Learning/telco_churn/")

if __name__ == "__main__":
    # data
    data = import_from_local(".")

    ## metadata variable management pipeline
    MetaDataManagement(data).metadata_management_pipeline()

    ### splitting
    X_train, X_valid, y_train, y_valid = train_valid_split(data)
    
    ## data variable management pipeline
    DataManagement(X_train).data_management_pipeline()
    DataManagement(X_valid).data_management_pipeline()


    # models pipelines
    ## random forest model
    evaluate_rdmf(X_train, X_valid, y_train, y_valid, n_estimators=20)

    # model comparaison