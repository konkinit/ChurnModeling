from pandas import Series


def quantiles_list(x: Series) -> list:
    list_quantile = [x.min(),
                     x.quantile(0.25),
                     x.quantile(0.5),
                     x.quantile(0.75),
                     x.max()]
    return list_quantile
