from Utils.Core import load_sav_to_dataframe, clip_dataset
from sklearn.model_selection import train_test_split
from easydict import EasyDict as edict
from sklearn.base import BaseEstimator, TransformerMixin
from preprocessing.Transformer.MissForest_sklearn import MissForestImputer
import inspect

class DataLoader(TransformerMixin, BaseEstimator):
    
    def __init__(self, data_path,settings:dict, random_state=None):
        """
        Initialize the DataLoader with dataset path and settings.
        Parameters:
        data_path (str): Path to the dataset file.
        settings (dict): Settings for data processing, including target column, test size, shuffle option, and possible targets.
        radnom_state (int, optional): Random state for reproducibility. Defaults to None.
        """
        self.settings = edict(settings.copy())
        self.dataset_path = data_path
        self.Random_state = random_state


    def Get_data(self, get_validation_set = False, split_data=True ,value_column_filter={}, one_class_mode=False, ):
        """
        Load the dataset and split it into training and testing sets.
        Parameters:
        get_validation_set (bool): If True, returns a validation set along with the training and testing sets.
        split_data: True to get splited data, False to get data without split. get_validation_set should be False when split_data True.
        value_column_filter: dictionary the key is column name and key_value is value to keep.
        Returns:
        tuple: A tuple containing the training and testing sets, and optionally the validation set.(x_train, x_test, y_train, y_test) or (x_train, x_val, x_test, y_train, y_val, y_test)
        """
        try:
            self.raw_data, _ = self.__load_data()

            self.__column_filter_by_value(value_column_filter)
            self.__box_plot_based_clip_outliers()
            self.__remove_ignored_column()
            
            if split_data:
                if get_validation_set:
                    return self.__split_data_train_validation_test(data=self.raw_data.copy(), test_size=self.settings.test_size, validation_size=0.25)
                return self.__split_data_train_test(data=self.raw_data.copy(), test_size=self.settings.test_size)
            else:
                if get_validation_set:
                    raise Exception(f"There is contradiction split_data parameter is {split_data} and get_validation_set is {get_validation_set}")
                return self.raw_data.drop(self.settings.target, axis=1) ,self.raw_data[self.settings.target]
        except:
            raise
    
    
    def __remove_ignored_column(self):
        """
        This function work inside of the Get_data function onlyl.
        """
        try:
            if self.settings.ignore_columns != None:
                self.raw_data = self.raw_data.drop(self.settings.ignore_columns, axis=1)
        except:
            raise
    def __box_plot_based_clip_outliers(self):
        """"
        This function work inside of the Get_data function onlyl.

        """
        try:
            if self.settings.box_plot_clipOutliers != []:
                self.raw_data = clip_dataset(self.raw_data, self.settings.box_plot_clipOutliers)
        except:
            raise
    
    def __column_filter_by_value(self,value_column_filter={}):
        """
        This function work inside of the Get_data function onlyl.
        This is responsiable to filter the data by column values
        Parameters:
        value_column_filter: dictionary the key is column name and key_value is value to keep.

        """
        #TODO: if it possible edit this function to only work on Get_data function
        try:
            if value_column_filter:
                for k, v in value_column_filter.items():
                    if k not in self.raw_data.columns.to_list():
                        raise Exception(f"The column {k} is not in column list: {self.raw_data.columns.to_list()}")
                    self.raw_data = self.raw_data[self.raw_data[f'{k}']== v]
                if len(self.raw_data) == 0:
                    raise Exception(f"No element has left after apply the filter {k}:{v}")
                

        except:
            raise
        pass
 
        
    def __load_data(self):
        try:
            return load_sav_to_dataframe(self.dataset_path)
        except:
            raise

    def __split_data_train_test(self, data, test_size=0.2):
        try:
            X, Y = data.drop(self.settings.target, axis=1), data[self.settings.target]
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, shuffle=self.settings.shuffle, random_state=self.Random_state)
            return (x_train, x_test, y_train, y_test)
        except:
            raise

    def __split_data_train_validation_test(self, data, test_size=0.2, validation_size=0.1):
        try:
            x_train, x_test, y_train, y_test = self.__split_data_train_test(data, test_size)
            x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=validation_size, shuffle=self.settings.shuffle, random_state=self.Random_state)
            return (x_train, x_val, x_test, y_train, y_val, y_test)
        except:
            raise



