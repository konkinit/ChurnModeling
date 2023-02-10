from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def evaluate_rdmf(
        X_train, 
        X_test, 
        y_train, 
        y_test, 
        n_estimators=20):
    """
    Train, fit and score the model 
    to test data to evaluate the model
    """
    rdmf = RandomForestClassifier(
        n_estimators=n_estimators)
    rdmf.fit(X_train, y_train)
    rdmf_score = rdmf.score(X_test, y_test)

    print("{} % of correct preds ".format(round(rdmf_score*100)))
    print("Confusion matrix")
    print(
        confusion_matrix
            (
            y_test, 
            rdmf.predict(X_test)
            )
        )