import pandas as pd
import numpy as np

def useless_feature(df):
    df.drop(["upsell_xsell", "issue_level2", "resolution", "city", "city_lat", "city_long", "data_usage_amt", "mou_onnet_6m_normal", "mou_roam_6m_normal", "region_lat","region_long", "state_lat", "state_long", "tweedie_adjusted"], axis=1, inplace=True)


def decode_char(df):
    list_vars_object = list(df.select_dtypes(exclude = ['int64', 'float64']).columns)
    for var in list_vars_object:
        df[var] = df[var].apply(lambda x : x.decode("utf-8"))

# some variables intuitively cannot have negative values, let replace negative value by 0 
def lower_limit(df):
    list_ = ["tot_mb_data_roam_curr", "seconds_of_data_norm", "lifetime_value", "bill_data_usg_m03", "bill_data_usg_m06", "voice_tot_bill_mou_curr",
            "tot_mb_data_curr", "mb_data_usg_roamm01", "mb_data_usg_roamm02", "mb_data_usg_roamm03", "mb_data_usg_m01", "mb_data_usg_m02", "mb_data_usg_m03",
            "calls_total", "call_in_pk", "calls_out_pk", "call_in_offpk", "calls_out_offpk", "mb_data_ndist_mo6m", "data_device_age",
            "mou_onnet_pct_MOM", "mou_total_pct_MOM"]
    for var in list_:
        df[var] = df[var].apply(lambda x : max(x, 0))


# handling the high skewness of the variables MB_Data_Usg_M by applying log transformation
def log_transform(df):
    for i in range(4, 10):
        df["log_MB_Data_Usg_M0"+str(i)] = df["MB_Data_Usg_M0"+str(i)].apply(lambda x: np.log(1+x))
        df.drop(columns=["MB_Data_Usg_M0"+str(i)])

# retrieving the list of variables having missing values
def missig_var(df):
    df_missing = df.isnull().sum().to_frame().reset_index()
    df_missing.columns = ["variable", "missing_nb"]
    df_missing = df_missing[df_missing["missing_nb"] > 0].reset_index(drop=True)
    df_missing['missing_pct'] = round(100 * df_missing['missing_nb'] / df.shape[0], 2)
    df_missing = df_missing.sort_values('missing_nb', ascending=False).reset_index(drop=True)
    return list(df_missing["variable"])

# missing indicator
def add_missing_indicator(df):
    list_missing_var = missing_var(df)
    df_ = df[list_missing_var].isnull().astype(int).add_suffix("_MI")
    df[df_.columns] = df_

# imputation
def imputation(df):
    list_missing_var = missing_var(df)
    for var in lis_missing_var:
        if len(df[df_val_mqtes.variable[1]].unique()) > 50:
            df[var].fillna(df[var].mean(), inplace=True)
        else :
            df[var].fillna(df[var].value_counts(ascending=False).to_frame().reset_index().iloc[0, 0], inplace=True)