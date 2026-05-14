from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationForestTrasnformer( ):
    def __init__(self, n_estimators, sample_size ,contamination=0.05, bootstrap=False,  max_features=1, verbose=0, random_state=None):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.sample_size = sample_size
        self.random_state = random_state
        
        self.model_ = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.sample_size,
            random_state=self.random_state,
            verbose=verbose, 
            bootstrap=bootstrap, 
            max_features=max_features
        )

    def fit(self, X, y=None):
        self.ok = True
        self.X_ = X.copy()
        self.model_.fit(X)
        return self

    def transform(self, X, y):
        if not self.ok:
            raise Exception("The imputer has not been fitted yet. Please call fit() before transform().")
        preds = self.model_.predict(X)
        mask = preds == 1
        self.mask_ = mask
        return X[mask], y[mask] if y is not None else None
        
    
    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X, y)

    def get_feature_names_out(self, input_features=None):
        if self.X_ is None:
            raise Exception("The model has not been fitted yet. Please call fit() before get_feature_names_out().")

        feature_names_out = self.X_.columns.to_list()
        return feature_names_out
    
