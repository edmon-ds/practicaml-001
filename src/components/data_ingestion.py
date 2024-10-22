import os
import sys

from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from sqlalchemy import create_engine
from dataclasses import dataclass

@dataclass
class DataIngestionConfig():
    #-------------------CSV PATHS-------------------------------------#
    raw_data_path:str = os.path.join("artifacts" , "dataset.csv")
    train_data_path:str = os.path.join("artifacts" , "trainset.csv")
    test_data_path:str = os.path.join( "artifacts", "testset.csv")


    #-------------------DATABASE CREDENTIALS-------------------------------------#
    driver:str = "ODBC+Driver+17+for+SQL+Server"
    server_name:str = "localhost"
    database:str = "BDdatasets"
    UID:str = "sa"
    PWD:str = "0440"

    connection_string:str = f"mssql+pyodbc://{UID}:{PWD}@{server_name}/{database}?driver={driver}"
    

class DataIngestion():
    def __init__(self):
        self.dataconfig = DataIngestionConfig() 
        
    def initate_data_ingestion(self):
        logging.info("enter the data ingestion method")
        try:
            engine = create_engine(self.dataconfig.connection_string)
            
            query = "Select * from Customers"

            df =  pd.read_sql_query(query , engine)
            
            df["SpendingScore"] = df["SpendingScore"] / 100   #se hace por que los modelos de regresion les cuesta mantenerse en un rango del 1 al 100
            
            logging.info("read the dataset as dataframe ")
        
            os.makedirs(os.path.dirname(self.dataconfig.raw_data_path ) , exist_ok=True)
            
            df.to_csv(self.dataconfig.raw_data_path , header=True)
            
            logging.info("se ha creado el csv")
            
            train_df , test_df = train_test_split(df , test_size = 0.2 , random_state = 42)
            
            logging.info("se ha separado el dataset y guardado")

            return (train_df , test_df)

        except Exception as e:
            raise CustomException(e , sys)