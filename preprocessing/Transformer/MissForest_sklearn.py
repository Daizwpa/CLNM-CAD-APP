from missforest import MissForest

from sklearn.base import BaseEstimator, TransformerMixin


class MissForestImputer(TransformerMixin, BaseEstimator):

    def __init__(self, max_iter=10, random_state=None):
        super().__init__()
        self.max_iter = max_iter
        self.random_state = random_state
        self.imputer_ = None

    def fit(self, X, y=None):
        self.X = X.copy()
        self.X = X.convert_dtypes()
        categoryal_columns = self.X.select_dtypes(include=["int"]).columns.to_list()
        self.imputer_ = MissForest(max_iter=self.max_iter, verbose=1 ,categorical= categoryal_columns )
        self.imputer_.fit(self.X)
        return self

    def transform(self, X):
        if self.imputer_ is None:
            raise Exception("The imputer has not been fitted yet. Please call fit() before transform().")
        output= self.imputer_.transform(X)
        output = output.convert_dtypes()
        return output

    def get_feature_names_out(self, input_features=None):
        if self.imputer_ is None:
            raise Exception("The imputer has not been fitted yet. Please call fit() before get_feature_names_out().")
    
        feature_names_out = self.imputer_.transform(self.X).columns.to_list()
        return feature_names_out

   

