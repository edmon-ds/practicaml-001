print()
from src.components.data_ingestion import * 
from src.components.data_transformation import *
from src.components.model_trainer import * 

if __name__ =="__main__":
    print("iniciando")
    print()
    
    print("creando rutas de acceso")
    print()

    data_ingestion = DataIngestion()
    train_df , test_df = data_ingestion.initate_data_ingestion()

    print("transformado datasets")
    print()
    data_transformation = DataTransformation()
    train_array , test_array = data_transformation.initiate_data_transformation(train_df , test_df)

    print("entrenando varios modelo")
    print()

    model_trainer = ModelTrainer()
    best_model_name ,best_model_score , best_model  = model_trainer.initiate_model_training(train_array , test_array)
    print("model entrenado y elegido")
    print()

    print("nombre del modelo y RMSE")
    print(f"nombre :  {best_model_name}  score : {best_model_score}")

    print("proceso finalizado")
    print()