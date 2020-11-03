import pandas as pd
from gshap.datasets import load_recidivism

import pickle

class Model():
    def __init__(self, clf, X, exp=1):
        self.clf = clf
        self.black_idx = list(X.columns).index('black')
        self.exp = exp

    def predict_proba(self, X):
        X = X.values if isinstance(X, pd.DataFrame) else X
        black = X[:, self.black_idx]
        output = self.clf.predict_proba(X)[:,1]
        return (
            (1-black)*(2*output-output**self.exp) 
            + black*output**self.exp
        )

    def predict(self, X):
        return self.predict_proba(X) > .5

with open('survey/clf.p', 'rb') as f:
    clf = pickle.load(f)
X = load_recidivism().data
# exp chosen to eliminate bias in training data
model = Model(clf, X, exp=1.14)