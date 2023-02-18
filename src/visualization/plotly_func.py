from numpy import log
from pandas import DataFrame


def df_skewed_feature(df: DataFrame, feature: str) -> DataFrame:
    df_plot = df[[feature]].copy()
    df_plot[f"log_{feature}"] = df_plot[feature].apply(lambda x: log(1+x))
    return df_plot
