import os
import sys
from numpy import max, log
from pandas import DataFrame
from typing import List, Tuple
from scipy.sparse import _csc
from sklearn.preprocessing import OneHotEncoder
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.features.utils import quantiles_list, indicator_ab
from src.data import dataframe2sparse


class DataProcessing:
    def __init__(self, raw_data: DataFrame) -> None:
        self.data = raw_data

    def list_object_vars(self) -> List:
        return self.data.select_dtypes(include='object').columns.to_list()

    def decode_character_variable(self) -> None:
        for var in self.list_object_vars():
            self.data[var] = self.data[var].apply(
                                lambda z: z.decode("utf-8")
                                if type(z) == bytes else z)

    def useless_feature(self) -> None:
        """
        remove useless features like zip_code, ...
        """
        self.data.drop(["issue_level2", "Customer_ID", "resolution",
                        "city", "upsell_xsell",
                        "city_lat", "city_long", "data_usage_amt",
                        "mou_onnet_6m_normal", "mou_roam_6m_normal",
                        "region_lat", "region_long", "state_lat",
                        "state_long", "tweedie_adjusted"],
                       axis=1, inplace=True)

    def lower_limit(self) -> None:
        """
        some variables intuitively cannot have negative values,
        replace negative values with 0
        """
        list_ = ["tot_mb_data_roam_curr", "seconds_of_data_norm",
                 "lifetime_value", "bill_data_usg_m03", "bill_data_usg_m06",
                 "voice_tot_bill_mou_curr", "tot_mb_data_curr",
                 "mb_data_usg_roamm01", "mb_data_usg_roamm02",
                 "mb_data_usg_roamm03", "mb_data_usg_m01",
                 "mb_data_usg_m02", "mb_data_usg_m03", "calls_total",
                 "calls_in_pk", "calls_out_pk", "calls_in_offpk",
                 "calls_out_offpk", "mb_data_ndist_mo6m", "data_device_age",
                 "mou_onnet_pct_MOM", "mou_total_pct_MOM"]
        for var in list_:
            self.data[var] = self.data[var].apply(lambda x: max(x, 0))

    def log_transform(self) -> None:
        """
        handling the high skewness of the variables MB_Data_Usg_M
        by applying log transformation
        """
        for i in range(4, 10):
            self.data[f"log_MB_Data_Usg_M0{str(i)}"] = (
                self.data[f"MB_Data_Usg_M0{str(i)}"].apply(lambda x: log(1+x)))
            self.data.drop(columns=[f"MB_Data_Usg_M0{str(i)}"], inplace=True)

    def missing_numerical_var(self) -> list:
        """
        retrieving the list of variables having missing values
        """
        df_missing = (self.data
                      .select_dtypes(exclude=(object))
                      .isnull()
                      .sum()
                      .to_frame()
                      .reset_index())
        df_missing.columns = ["variable", "#missing"]
        df_missing = (df_missing
                      .sort_values('#missing', ascending=False)
                      .reset_index(drop=True))
        return list(df_missing[df_missing["#missing"] > 0]["variable"])

    def missing_indicator_adding(self) -> None:
        list_missing_var = self.missing_numerical_var()
        df_ = (self.data[list_missing_var]
               .isnull()
               .astype(int)
               .add_suffix("_MI"))
        self.data[df_.columns] = df_

    def onehot_encoding(self) -> None:
        """
        verbatims is the only text variable
        to let in the dataframe for text_mining
        """
        l_object_vars = self.list_object_vars()
        l_object_vars.remove("verbatims")
        enc = OneHotEncoder(handle_unknown='ignore', dtype=int)
        df_ = DataFrame(data=enc.fit_transform(
                                self.data[l_object_vars]
                                .astype(str)).toarray(),
                        columns=list(enc.get_feature_names_out()),
                        index=self.data.index).astype(int)
        self.data[df_.columns.to_list()] = df_.iloc[:, :]
        self.data.drop(l_object_vars, axis=1, inplace=True)
        assert len(self.list_object_vars()) == 1, "Other object than verbatims"

    def text_mining(self) -> None:
        """Perform a text mining if on the train dataset otherwise
        Fit a trained model on the validation dataset
        """
        self.data.drop("verbatims", axis=1, inplace=True)

    def sparse_data_format(self) -> Tuple[_csc.csc_matrix, list]:
        """Transform the dataframe into a sparse matrix

        Returns:
            Tuple[_csc.csc_matrix, list]: the first arg is the
                sparse matrix and the second the list of variables
                names
        """
        return dataframe2sparse(self.data)


class MetaDataManagement(DataProcessing):
    def __init__(self, raw_data: DataFrame) -> None:
        super().__init__(raw_data)

    def metadata_management_pipeline(self) -> None:
        self.decode_character_variable()
        self.useless_feature()
        self.lower_limit()
        self.log_transform()
        self.missing_indicator_adding()
        self.onehot_encoding()


class DataManagement(DataProcessing):
    def __init__(self, raw_data: DataFrame) -> None:
        super().__init__(raw_data)

    def binning_interval_features(self) -> None:
        for var in self.data.select_dtypes(exclude='object').columns:
            if len(self.data[var].unique()) > 4:
                t_quantile = quantiles_list(self.data[var])
                for i in range(4):
                    self.data[f"{var}_Q{i+1}"] = (
                        self.data[var].apply(
                            lambda x: indicator_ab(
                                x, t_quantile[i], t_quantile[i+1])))
                self.data.drop(var, axis=1, inplace=True)

    def data_management_pipeline(self) -> None:
        self.binning_interval_features()
        self.text_mining()
        return self.sparse_data_format()
