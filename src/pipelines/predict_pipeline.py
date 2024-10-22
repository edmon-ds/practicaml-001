from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig

from src.utils import * 
import pandas as pd


class CustomData:
    def __init__(self , Gender:str	, Age:int , 	AnnualIncome:int  ,	Profession:str,	WorkExperience:int,	FamilySize:int):
        
        self.data_user = pd.DataFrame({
        "Gender":[Gender] , 
        "Age":[Age ], 
        "AnnualIncome":[AnnualIncome] ,  
        "Profession":[Profession] , 
        "WorkExperience":[WorkExperience] , 
        "FamilySize":[FamilySize]  
        })

    
    def get_data_as_dataframe(self):
        try:
            return self.data_user
        except Exception as e:
            raise CustomException(e , sys)


class PredictPipeline():
    def __init__(self):

        self.preprocessor = load_object(DataTransformationConfig().preprocessor_obj_file_path)
        self.model = load_object(ModelTrainerConfig().trained_model_file_path)
    
    def predict(self , data_user):
        try:
            data_transformed = self.preprocessor.transform(data_user)
            preds = self.model.predict(data_transformed)
            return preds
        except Exception as e:
            raise CustomException(e , sys)