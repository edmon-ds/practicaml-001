import os
import sys

from src.logger import logging
from src.exception import CustomException

import pandas as pd
from sklearn.model_selection import train_test_split

from sqlalchemy import create_engine
from data_config import DataConfig

class DataIngestion():
    def __init__(self):
        self.dataconfig = DataConfig() 
        
    def initate_data_ingestion(self):
        logging.info("enter the data ingestion method")
        try:
            engine = create_engine(self.dataconfig.connection_string)
            
            query = "Select * from Customers"

            df =  pd.read_sql_query(query , engine)
            
            logging.info("read the dataset as dataframe ")
        
            os.makedirs(os.path.dirname(self.dataconfig.raw_data_path ) , exist_ok=True)
            
            df.to_csv(self.dataconfig.raw_data_path)
            
            logging.info("se ha creado el csv")
            
            train_df , test_df = train_test_split(df , test_size = 0.2 , random_state = 42)
            
            train_df.to_csv(self.dataconfig.train_data_path , index = False , header = False )
            test_df.to_csv(self.dataconfig.test_data_path , index = False , header = False )
            
            logging.info("se ha separado el dataset")
            return 0


        except Exception as e:
            raise CustomException(e , sys)

print(DataIngestion().initate_data_ingestion())