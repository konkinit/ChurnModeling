from sklearn.model_selection import train_test_split

def make_val_split(df):
    y = df["churn"]
    X = df.drop("churn", axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test

def train_test_data(X_train, X_test, y_train, y_test):
    train_data = X_train
    train_data["churn"] = y_train
    test_data = X_test
    test_data["churn"] = y_test
    return train_data, test_data