import __init__
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder, MinMaxScaler, FunctionTransformer, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import numpy as np
from easydict import EasyDict as edict
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing.Transformer.MissForest_sklearn import MissForestImputer

class PreprocessingPipeline(TransformerMixin, BaseEstimator):
    """
    For coding, scaling and imputing dataset 
    """
    def __init__(self, settings:dict, random_state=None):
        self.settings = edict(settings.copy())
        self.random_state = random_state
        self.ignored_columns = [*self.settings.ignore_columns, *self.settings.target]
        self.pipeline_ = self.__Compose_pipeline()

    def fit(self, X, y=None):
        try:
            self.X_ = X.copy()
            self.pipeline_.fit(X)
            return self
        except:
            raise

    def transform(self, X):
        try:
            if self.pipeline_ == None:
                raise Exception("The imputer has not been fitted yet. Please call fit() before transform().")
            output = self.pipeline_.transform(X)
            output =output.convert_dtypes()
            
            cat = list(output.select_dtypes("Int64").columns)
            output[cat] = output[cat].astype("int8")
            
            return output
        except:
            raise

    def __Compose_pipeline(self):
        try: 
            return Pipeline( [
                (
                    "coding", 
                    ColumnTransformer(
                        [
                            ("nominal", self.__Nominal_pipeline(), self.__get_nominal_columns()),
                            ("ordinal", self.__Ordinal_pipeline(), self.__get_ordinal_columns()),
                            ("binary", self.__Binary_pipeline(), self.__get_binary_columns()),
                            ("numerical", self.__Scaling_pipeline(), self.__get_numerical_columns()),
                        ]
                    )
                 ),
                (
                    "imputating",
                    MissForestImputer(max_iter=self.settings.inputing_max_iter,random_state=self.random_state)
                )
            
            ]).set_output(transform="pandas")
        except:
            raise

    def __Nominal_pipeline(self):
        try:
            cat = self.__get_nominal_possible_values()
            return OneHotEncoder(categories= cat, sparse_output=False) 
        except:
            raise 

    def __Ordinal_pipeline(self):
        try:
            return OrdinalEncoder( handle_unknown='use_encoded_value',
                                            unknown_value=np.nan,
                                            categories=self.__get_value_order_of_ordinal_columns())
                
        except:
            raise

    def __Binary_pipeline(self):
        try:
            return  OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan)
        except:
            raise

    def __Scaling_pipeline(self):
        try:
            return Pipeline([
                #("quntail", QuantileTransformer(n_quantiles=10)),
                ( "Scaler", MinMaxScaler(feature_range=(-1, 1)) )
            ])
        except:
            raise

    def __get_nominal_columns(self):
        return [ col for col in list(self.settings.nominal_columns.keys()) if col not in self.ignored_columns]

    def __get_nominal_possible_values(self):
        return [val for key, val  in self.settings.nominal_columns.items() if key not in self.ignored_columns]

    def __get_value_order_of_ordinal_columns(self):
        return [val for key, val  in self.settings.ordinal_columns.items() if key not in self.ignored_columns]

    def __get_ordinal_columns(self):
        return [col for col in list(self.settings.ordinal_columns.keys()) if col not in self.ignored_columns]

    def __get_binary_columns(self):
        return [col for col in self.settings.binary_columns if col not in self.ignored_columns]

    def __get_numerical_columns(self):
        return [col for col in self.settings.numerical_columns if col not in self.ignored_columns]

    def get_feature_names_out(self, input_features=None):
        if self.X_ is None:
            raise Exception("The model has not been fitted yet. Please call fit() before get_feature_names_out().")

        feature_names_out = self.transform(self.X_).columns.to_list()
        return feature_names_out

