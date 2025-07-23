# inference_utils.py

import joblib

class FraudModelPreprocessor:
    def __init__(self):
        self.columns = []

    def fit(self, df, target_col='isFraud'):
        self.columns = [col for col in df.columns if col != target_col]

    def transform(self, df):
        return df[self.columns]

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
