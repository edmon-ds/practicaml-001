from flask import Flask , request , render_template

from src.pipelines.predict_pipeline import CustomData , PredictPipeline
from src.exception import CustomException
from src.logger import logging
import sys


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict" , methods = ["GET" ,"POST" ]  )
def predict_datapoint():
    if request.method =="GET":
        return render_template("predict.html")
    else:
        try:
            data_raw = CustomData(  Gender =  request.form.get("Gender")  , 
            Age =  request.form.get("Age")  , 
            AnnualIncome = request.form.get("AnnualIncome")  ,  
            Profession = request.form.get("Profession") , 
            WorkExperience = request.form.get("WorkExperience")   , 
            FamilySize =  request.form.get("FamilySize")  
            )
            data_df = data_raw.get_data_as_dataframe()
            predict_pipeline = PredictPipeline()
            preds = predict_pipeline.predict(data_df)
            
            return render_template("predict.html" , results = preds[0])

        except Exception as e:
            raise CustomException(e , sys)

if __name__ =="__main__":
    app.run(host= "0.0.0.0", port =8080 , debug=True)