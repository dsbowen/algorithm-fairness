import gshap
import pandas as pd
from gshap.datasets import load_recidivism

import pickle

class Model():
    def __init__(self, clf, X, exp=1):
        self.clf = clf
        self.columns = list(X.columns)
        self.black_idx = self.columns.index('black')
        self.exp = exp
        self.base_rate = self.predict_proba(X).mean()
        self.explainer = gshap.KernelExplainer(
            self.predict_proba, X, g=lambda x: x
        )

    def predict_proba(self, X):
        X = X.values if isinstance(X, (pd.DataFrame, pd.Series)) else X
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        black = X[:, self.black_idx]
        output = self.clf.predict_proba(X)[:,1]
        return (
            (1-black)*(2*output-output**self.exp) 
            + black*output**self.exp
        )

    def predict(self, X):
        return self.predict_proba(X) > .5

    def shap_values(self, X, output=None, nsamples=32):
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
        shap_values = self.explainer.gshap_values(X, nsamples=nsamples)
        # normalize to account for sampling error
        output = self.predict_proba(X) if output is None else output
        shap_values *= (output - self.base_rate) / shap_values.sum(axis=0)
        return pd.DataFrame(shap_values.T, columns=self.columns)


with open('survey/clf.p', 'rb') as f:
    clf = pickle.load(f)
X = load_recidivism().data
X = X.drop(columns='high_supervision')
# exp chosen to eliminate bias in training data
model = Model(clf, X, exp=1.14)