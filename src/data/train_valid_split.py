from sklearn.model_selection import train_test_split

def train_valid_split(df):
    y = df["churn"]
    X = df.drop("churn", axis=1)
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                X,
                                                y, 
                                                test_size=0.30, 
                                                stratify=y, 
                                                random_state=42)
    return X_train, X_valid, y_train, y_valid
