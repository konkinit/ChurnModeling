from pandas import Series
from numpy import log
from pandas import DataFrame


def quantiles_list(x: Series) -> list:
    list_quantile = [x.min(),
                     x.quantile(0.25),
                     x.quantile(0.5),
                     x.quantile(0.75),
                     x.max()]
    return list_quantile


def indicator_ab(x: float, a: float, b: float):
    if a-0.001 < x <= b:
        return 1
    return 0


def df_skewed_feature(df: DataFrame, feature: str) -> DataFrame:
    df_plot = df[[feature]].copy()
    df_plot[f"log_{feature}"] = df_plot[feature].apply(lambda x: log(1+x))
    return df_plot
