# # train_model.py
# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
# import joblib
# import os

# def train_model(input_path="data/kaggle_fraud_processed.csv", output_path="models/fraud_model.joblib"):
#     df = pd.read_csv(input_path)

#     X = df.drop("isFraud", axis=1)
#     y = df["isFraud"]

#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#     clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
#     clf.fit(X_train, y_train)

#     y_pred = clf.predict(X_test)
#     print(classification_report(y_test, y_pred))

#     os.makedirs("models", exist_ok=True)
#     joblib.dump(clf, output_path)
#     print(f"✅ Saved model to {output_path}")

# if __name__ == "__main__":
#     train_model()

# train_model.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

def train_model(input_path="data/kaggle_fraud_processed.csv", output_path="models/fraud_model.joblib"):
    df = pd.read_csv(input_path)

    X = df.drop("isFraud", axis=1)
    y = df["isFraud"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    os.makedirs("models", exist_ok=True)
    # CHANGED: Use compress parameter without protocol for better compatibility
    joblib.dump(clf, output_path, compress=3)
    print(f"✅ Saved model to {output_path}")

if __name__ == "__main__":
    train_model()