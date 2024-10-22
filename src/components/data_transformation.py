import sys

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder , StandardScaler

from src.exception import CustomException
from src.logger import logging

from src.utils import *


from dataclasses import dataclass
from sklearn.model_selection import train_test_split


@dataclass
class DataTransformationConfig():
    preprocessor_obj_file_path:str  = os.path.join("artifacts" , "preprocessor.pkl" )

class DataTransformation():
    def __init__(self):
        self.dataconfig = DataTransformationConfig()
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
    
       
    def initiate_data_transformation(self , train_df , test_df):
        try:
            
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

            save_object(file_path=self.dataconfig.preprocessor_obj_file_path , obj = preprocessor )
            
            return (train_array , test_array)

        except Exception as e:
            raise CustomException(e, sys)