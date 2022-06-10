from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

def evaluate_rdmf(X_train, X_test, y_train, y_test, n_estimators=20):
    # Train
    rdmf = RandomForestClassifier(n_estimators=n_estimators)
    rdmf.fit(X_train, y_train)

    # Fit
    rdmf_score = rdmf.score(X_test, y_test)

    print("{} % de bonnes réponses sur les données de test pour validation".format(round(rdmf_score*100)))
    print("Confusion matrix")
    print(confusion_matrix(y_test, rdmf.predict(X_test)))