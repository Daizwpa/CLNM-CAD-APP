import mlflow
from mlflow.data.pandas_dataset import PandasDataset
from abc import ABC, abstractmethod

from mlflow.models import infer_signature
import warnings


class recorder_mlflow(ABC):
    def __init__(self, experiment_name, server_url="http://127.0.0.1:5000"):
        self.experiment_name = experiment_name
        self.server_url = server_url
        self._dataset ={}
        self._artifcates =[]

    def SetDataset(self, x, y, set_name, target):
        try:
            if set_name in self._dataset:
                warnings.warn(f"the set {set_name} already existed")
            data = x.copy()
            data[target] = y.copy()
            data :PandasDataset = mlflow.data.from_pandas(data, targets=target)
            self._dataset[set_name] = data
            self._artifact_paths = []
        except:
            raise
    
    def __log_Artifactrtificats(self):
        try:
            for artificate_path in self._artifact_paths:
                mlflow.log_artifact(artificate_path)

        except:
            raise
    
    def SetArtifactrtificats(self, artifact_path:str):
        try:
            self._artifact_paths.append(artifact_path)
        except:
            raise
    
    def __get_input_example(self):
        try:
            if len(self._dataset) == 0:
                Exception(f"set the dataset first with {mlflow.SetDataset.__name__}")
            data = list(self._dataset.items())[0][1].df
            target_name = list(self._dataset.items())[0][1].targets
            x = data.drop([target_name], axis=1)
            return x
        except:
            raise

    def __log_datasets(self):
        try:
            for set_name, set in self._dataset.items():
                mlflow.log_input(set, context=set_name)
        except:
            raise
    
    def __get_signature(self):
        try:
            if len(self._dataset) == 0:
                Exception(f"set the dataset first with {mlflow.SetDataset.__name__}")
            data = list(self._dataset.items())[0][1].df
            target_name = list(self._dataset.items())[0][1].targets
            x = data.drop([target_name], axis=1)
            y = data[[target_name]]
            return infer_signature(x, y)
        except:
            raise
    
    @abstractmethod
    def _record_model(self, model, artificat_name, signature, input_example):
        pass

    def Record(self, model, name, param, metrics, description):
        try:
            mlflow.set_experiment(self.experiment_name)
            mlflow.set_tracking_uri(uri=self.server_url)
            
            with mlflow.start_run(run_name=name, description=description):
                
                mlflow.log_params(param)
                mlflow.log_metrics(metrics)
                self.__log_datasets()

                signature = self.__get_signature()
                input_example = self.__get_input_example()
                self.__log_Artifactrtificats()#

                mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=type(model).__name__,
                    signature= signature,
                    input_example= self.__get_input_example(),
                ) 

                self._record_model(model,type(model).__name__, signature, input_example)
        except:
            raise




class recorder_mlflow_sklearn(recorder_mlflow):
    def _record_model(self, model, artificat_name, signature, input_example):
        try:
            mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=artificat_name,
                    signature= signature,
                    input_example= input_example,
            ) 
        except:
            raise

class recorder_mlflow_xgboost(recorder_mlflow):
    def _record_model(self, model, artificat_name, signature, input_example):
        try:
            mlflow.xgboost.log_model(
                xgb_model=model,
                artifact_path=artificat_name,
                signature= signature,
                input_example= input_example,
            )
        except:
            raise