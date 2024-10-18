import sys
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import *

from data_config import DataConfig
class DataTransformation():
    def __init__(self):
        self.dataconfig = DataConfig()
        self.numerical_features = ['Age', 'AnnualIncome', 'WorkExperience', 'FamilySize']
        self.categorical_features = [ 'Gender', 'Profession']
        self.label = 'SpendingScore'
 
    def get_preprocessor(self):
        '''this function create the preprocessor'''
        try:
            num_pipeline = Pipeline(
                steps = [
                        ("imputer" , SimpleImputer(strategy ="mean") ) , 
                        ("scaler" , StandardScaler())
                        ]
            )
            cat_pipeline = Pipeline(
                steps = [
                            ("imputer", SimpleImputer(strategy='most_frequent')) , 
                            ("one_hot_encoder" , OneHotEncoder())
                ]
            )
            preprocessor = ColumnTransformer(
                [
            ("num_pipeline" , num_pipeline, self.numerical_features ),
            ("cat_pipeline" , cat_pipeline ,self.categorical_features )
                ]
            )

            return preprocessor

        except Exception as e:
            CustomException(e , sys)
    
    def get_data_transformed(self):
        try:
            train_df = pd.read_csv(DataConfig.train_data_path)
            test_df = pd.read_csv(DataConfig.test_data_path)

            logging.info("train and test set readed")
            

            preprocessor = self.get_preprocessor()

            train_input_raw = train_df.drop(columns = [self.label])
            train_label = train_df[self.label]

            test_input_raw = test_df.drop(columns = [self.label])
            test_label = test_df[self.label]


            logging.info("applying preprocessing to the dataset")

            train_input_array = preprocessor.fit_transform(train_input_raw)
            test_input_array = preprocessor.transform(test_input_raw)

            train_array = np.c_[train_input_array , train_label.values]
            test_array = np.c_[test_input_array , test_label.values]

            logging.info("saving preprocessing")

            save_object(file_path=self.dataconfig.preprocessor_obj_file_path)
            
            return (train_array , test_array)

        except Exception as e:
            raise CustomException(e, sys)