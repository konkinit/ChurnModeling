from numpy import nan
from pandas import DataFrame, concat
from typing import Dict, Any


num_keys = [
                "Variable Name", "Label", "Type",
                "Number of missing",
                "Number of distinct values",
                "Minimum", "Maximum", 
                "Mean", "Mode",
                "Standard Deviation", "Skewness", "Kurtosis"
            ]


char_keys = [
                "Variable Name", "Label", "Type",
                "Number of missing",
                "Number of categories", "Mode"
            ]


class MetadataStats:
    def __init__(
            self, 
            raw_data: DataFrame
        ) -> None:
        self.data = raw_data

    
    def variable_summarizing(
            self,
            var: str) -> DataFrame:
        """
        Summarize a given variable
        """
        if self.data[var].dtype != 'object':
            if len(self.data[var].unique()) > 50:
                l_values = [
                    var, var, "Interval",
                    self.data[var].isna().sum(), nan,
                    round(self.data[var].min(), 3), round(self.data[var].max(), 3), 
                    round(self.data[var].mean(), 3), round(self.data[var].mode()[0], 3), 
                    round(self.data[var].std(), 3), 
                    round(self.data[var].skew(), 3), round(self.data[var].kurt(), 3)]
                return DataFrame.from_dict([{num_keys[i]: l_values[i] for i in range(12)}])
                
            else:
                l_values = [
                    var, var, "Ordinal",
                    self.data[var].isna().sum(), len(self.data[var].unique()),
                    self.data[var].min(), self.data[var].max(), 
                    round(self.data[var].mean(), 3), self.data[var].mode(),
                    nan, 
                    nan, nan]
                return DataFrame.from_dict([{num_keys[i]: l_values[i] for i in range(12)}])
        else:
            if len(self.data[var].unique()) > 1000:
                l_values = [
                    var, var, "Text",
                    self.data[var].isna().sum(), nan, nan]
                return DataFrame.from_dict([{char_keys[i]: l_values[i] for i in range(6)}])
            else:
                l_values = [
                    var, var, "Categorical",
                    self.data[var].isna().sum(), len(self.data[var].unique()),
                    self.data[var].mode().values[0]]
                return DataFrame.from_dict([{char_keys[i]: l_values[i] for i in range(6)}])
        
    
    def metadata_report(
            self,
            vars_type: str
            ) -> DataFrame:

        if vars_type == 'char':
            df = DataFrame(columns=char_keys)
            l_char_vars = (self.data
                        .select_dtypes(include='object')
                        .columns
                        .tolist())
        
        else: 
            df = DataFrame(columns=num_keys)
            l_char_vars = (self.data
                        .select_dtypes(exclude='object')
                        .columns
                        .tolist())
        
        for var in l_char_vars:
            df = concat(
                        [df, self.variable_summarizing(var)], 
                        axis=0, 
                        ignore_index=True)
        return df.sort_values("Type").reset_index(drop=True)
