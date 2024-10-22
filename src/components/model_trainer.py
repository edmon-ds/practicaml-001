import sys
from src.logger import logging
from src.exception import CustomException

from xgboost import XGBRegressor
from sklearn.ensemble import AdaBoostRegressor 
from sklearn.svm import SVR

from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import mean_absolute_error

from sklearn.model_selection import GridSearchCV

import pandas as pd

from src.utils import *
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig():
    #-------------------Model PATH-------------------------------------#
    trained_model_file_path:str = os.path.join("artifacts","model.pkl")


class ModelTrainer():
    def __init__(self):
        self.dataconfig = ModelTrainerConfig()
        
        self.models = {
                        "xgb" :XGBRegressor(n_estimators =  40) , 
                        "AdaBoostRegressor" : AdaBoostRegressor(n_estimators =  40) , 
                        "SVR": SVR()
                       }
        
        self.params = {
                        "xgb": {
                        "max_depth": [3, 5, 7]  # Opciones para el hiperparámetro max_depth
                        },
                        "AdaBoostRegressor": {
                        "learning_rate": [0.01, 0.1, 1]  # Opciones para el hiperparámetro learning_rate
                        },
                        "SVR": {
                        "C": [0.1, 1, 10]  # Opciones para el hiperparámetro C
                         }

                         }
        
        self.best_model = None
        self.best_model_name = ""
        self.best_model_score = "" #almacenara el RMSE
        self.model_threshold = 0.6 #almacenara el threshold que no debe pasar el RMSE

    def initiate_model_training(self ,train_array , test_array ):
        try:
            logging.info("split training and test set")
            
            X_train , y_train , X_test  , y_test = (train_array[: , :-1] , train_array[: , -1] , test_array[: , :-1] , test_array[: , -1] ) 
            
            model_report_df = self.evaluate_models(X_train , y_train , X_test  , y_test)

            best_model_row = model_report_df.loc[model_report_df["RMSE"].idxmin()]
            self.best_model_name = best_model_row["model"]
            self.best_model_score = best_model_row["RMSE"]
            
            self.best_model = self.models[self.best_model_name]

            if self.best_model_score >= self.model_threshold:
                raise CustomException("the model predict is poor")
            
            logging.info("best modelo found") 

            save_object(self.dataconfig.trained_model_file_path , obj= self.best_model)
            
            return (self.best_model_name ,self.best_model_score , self.best_model )
              
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_models(self , X_train , y_train , X_test , y_test):
        try:
            report = []
            for model_name , model in self.models.items():
                params = self.params[model_name]
                gs = GridSearchCV(model , params , cv = 3)
                gs.fit(X_train,y_train)

                model.set_params(**gs.best_params_)

                model.fit(X_train , y_train)

                y_test_pred = model.predict(X_test)

                report.append({"model":model_name , 
                               "RMSE": root_mean_squared_error( y_test,y_test_pred ) , 
                               "MAE":mean_absolute_error(y_test,y_test_pred ) } ) 
            
            return pd.DataFrame( report )

        except Exception as e: 
            raise CustomException(e , sys)      