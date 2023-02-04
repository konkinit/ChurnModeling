from numpy import max, log
from pandas import (
    DataFrame, concat, get_dummies)
from typing import List
from sklearn.preprocessing import OneHotEncoder


class DataProcessing:
    def __init__(self, raw_data: DataFrame) -> None:
        self.data = raw_data


    def useless_feature(self) -> None:
        """
        remove useless features like zip_code, ...
        """
        self.data.drop(["issue_level2", "resolution", "city", 
                    "city_lat", "city_long", "data_usage_amt", 
                    "mou_onnet_6m_normal", "mou_roam_6m_normal", 
                    "region_lat", "region_long", "state_lat", 
                    "state_long", "tweedie_adjusted", "upsell_xsell"], 
                    axis=1, inplace=True)


    def decode_char(self) -> None:
        """
        decode str to UTF-8
        """
        list_vars_object = list(self.data.select_dtypes(
                                exclude=['int64', 'float64']
                                ).columns)
        for var in list_vars_object:
            self.data[var] = self.data[var].apply(lambda x : x.decode("utf-8"))


    def lower_limit(self) -> None:
        """
        some variables intuitively cannot have negative values, 
        replace negative values with 0 
        """
        list_ = ["tot_mb_data_roam_curr", "seconds_of_data_norm", "lifetime_value", 
                "bill_data_usg_m03", "bill_data_usg_m06", "voice_tot_bill_mou_curr",
                "tot_mb_data_curr", "mb_data_usg_roamm01", "mb_data_usg_roamm02", 
                "mb_data_usg_roamm03", "mb_data_usg_m01", "mb_data_usg_m02", 
                "mb_data_usg_m03", "calls_total", "calls_in_pk", "calls_out_pk", 
                "calls_in_offpk", "calls_out_offpk", "mb_data_ndist_mo6m", "data_device_age",
                "mou_onnet_pct_MOM", "mou_total_pct_MOM"]
        for var in list_:
            self.data[var] = self.data[var].apply(lambda x : max(x, 0))


    def log_transform(self) -> None:
        """
        handling the high skewness of the variables MB_Data_Usg_M 
        by applying log transformation
        """
        for i in range(4, 10):
            self.data[f"log_MB_Data_Usg_M0{str(i)}"] = (self.data[f"MB_Data_Usg_M0{str(i)}"]
                                                            .apply(lambda x: log(1+x))
                                                        )
            self.data.drop(columns=[f"MB_Data_Usg_M0{str(i)}"], inplace=True)


    def missing_var(self) -> List:
        """
        retrieving the list of variables having missing values
        """
        df_missing = (self.data
                        .isnull()
                        .sum()
                        .to_frame()
                        .reset_index()
                    )   
        df_missing.columns = ["variable", "missing_nb"]
        df_missing = df_missing[df_missing["missing_nb"] > 0].reset_index(drop=True)
        df_missing = (df_missing
                        .sort_values('missing_nb', ascending=False)
                        .reset_index(drop=True)
                    )
        return list(df_missing["variable"])

    
    def add_missing_indicator(self) -> None:
        """
        create missing indicator for features with missingness
        """
        list_missing_var = self.missing_var()
        df_missing_indicator = (self.data[list_missing_var]
                                    .isnull()
                                    .astype(int)
                                    .add_suffix("_MI")
                                )
        self.data = concat([self.data, df_missing_indicator], axis=1)
    

    def imputation(self):
        """
        impute missing values with right method
        """
        list_missing_var = self.missing_var()
        assert (
            len(list_missing_var) > 0
        ), "Columns with missing values"
        for var in list_missing_var:
            if len(self.data[var].unique()) > 50:
                """
                condition that a variable is continious
                """
                self.data[var].fillna(self.data[var].mean(), inplace=True)
            else :
                self.data[var].fillna(
                    self.data[var].value_counts(ascending=False
                    ).to_frame().reset_index().iloc[0, 0], inplace=True)
    
    def list_object_vars(self) -> List:
        return self.data.select_dtypes(include='object').columns.to_list()

    def onehot_encoding(self) -> None:
        """
        verbatims is the only tet variable 
        to let in the dataframe for text_mining
        """
        l_object_vars = self.list_object_vars()
        enc = OneHotEncoder(handle_unknown='ignore')
        df_ = DataFrame(data=enc.fit_transform(self.data[l_object_vars]).toarray(), 
            columns=list(enc.get_feature_names_out())).astype(int)
        self.data.drop(l_object_vars, axis=1, inplace=True)
        self.data = concat([self.data, df_], 
                            axis=1)
        assert (len(self.list_object_vars()) == 0), "No character columns in data"

    
    def text_mining(self) -> None:
        """
        remove the variable to process with text mining methods
        until the right techniques are found
        """
        self.data.drop("verbatims", axis=1, inplace=True)


class MetaDataManagement(DataProcessing):
    def __init__(self, raw_data: DataFrame) -> None:
        super().__init__(raw_data)

    def metadata_management_pipeline(self):
        self.useless_feature()
        self.decode_char()
        self.lower_limit()
        self.log_transform()
        self.add_missing_indicator()


class DataManagement(DataProcessing):
    def __init__(self, raw_data: DataFrame) -> None:
        super().__init__(raw_data)

    def data_management_pipeline(self):
        self.imputation()
        self.text_mining()
        self.onehot_encoding()
        
