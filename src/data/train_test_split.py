from sklearn.model_selection import train_test_split

def train_test_splitting(df):
    y = df["churn"].values
    X = df.drop("churn", axis=1).values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
    return X_train, X_test, y_train, y_test