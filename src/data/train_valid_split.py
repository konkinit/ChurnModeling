from sklearn.model_selection import train_test_split
from pandas import DataFrame


def train_valid_splitting(df: DataFrame, frac: float):
    y = df["churn"]
    X = df.drop("churn", axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                X,
                                                y,
                                                test_size=1-frac,
                                                stratify=y,
                                                random_state=42)
    return X_train, X_valid, y_train, y_valid
