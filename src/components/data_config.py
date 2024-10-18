from dataclasses import dataclass
import os

@dataclass
class DataConfig:
    #-------------------CSV PATHS-------------------------------------#
    raw_data_path:str = os.path.join("artifacts" , "dataset.csv")
    train_data_path:str = os.path.join("artifacts" , "trainset.csv")
    test_data_path:str = os.path.join( "artifacts", "testset.csv")

    #--------------------------------------------------------------#

    #-------------------DATABASE CREDENTIALS-------------------------------------#
    driver:str = "ODBC+Driver+17+for+SQL+Server"
    server_name:str = "localhost"
    database:str = "BDdatasets"
    UID:str = "sa"
    PWD:str = "0440"

    connection_string:str = f"mssql+pyodbc://{UID}:{PWD}@{server_name}/{database}?driver={driver}"
    
    #--------------------------------------------------------------#

    #-------------------PREPROCESSOR PATH-------------------------------------#
    preprocessor_obj_file_path  = os.path.join("artifacts" , "preprocessor.pkl" )

