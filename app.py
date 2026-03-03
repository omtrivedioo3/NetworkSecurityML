import sys
import os

import certifi
ca = certifi.where()

from dotenv import load_dotenv
load_dotenv()
mongo_db_url = os.getenv("MONGODB_URL_KEY")
print(mongo_db_url)
import pymongo
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.pipeline.training_pipeline import TrainingPipeline
from fastapi.staticfiles import StaticFiles
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, "networksecurity", "templates")
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, File, UploadFile,Request
from uvicorn import run as app_run
from fastapi.responses import Response
from starlette.responses import RedirectResponse
import pandas as pd

from networksecurity.utils.main_utils.utils import load_object

from networksecurity.utils.ml_utils.model.estimator import NetworkModel


client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)

from networksecurity.constant.training_pipeline import DATA_INGESTION_COLLECTION_NAME
from networksecurity.constant.training_pipeline import DATA_INGESTION_DATABASE_NAME

database = client[DATA_INGESTION_DATABASE_NAME]
collection = database[DATA_INGESTION_COLLECTION_NAME]

app = FastAPI()
origins = ["*"]

STATIC_DIR = os.path.join(BASE_DIR, "networksecurity", "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.templating import Jinja2Templates


templates = Jinja2Templates(directory=TEMPLATE_DIR)

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.get("/train")
async def train_route():
    try:
        train_pipeline=TrainingPipeline()
        metrics =train_pipeline.run_pipeline()
        return {
            "status": "Training Completed Successfully",
            "model_performance": metrics
        }
    except Exception as e:
        raise NetworkSecurityException(e,sys)
    
@app.post("/predict")
async def predict_route(request: Request,file: UploadFile = File(...)):
    try:
        df=pd.read_csv(file.file)
        #print(df)
        preprocesor=load_object("final_model/preprocessor.pkl")
        final_model=load_object("final_model/model.pkl")
        network_model = NetworkModel(preprocessor=preprocesor,model=final_model)
        print(df.iloc[0])
        y_pred = network_model.predict(df)

        # Convert numeric prediction to meaningful label
        df['Prediction'] = y_pred
        df['Prediction'] = df['Prediction'].map({
            1: "Legitimate Website",
            0: "Phishing Website",
            -1: "Phishing Website"
        })

        # Summary Counts
        total_records = len(df)
        phishing_count = (df['Prediction'] == "Phishing Website").sum()
        legitimate_count = (df['Prediction'] == "Legitimate Website").sum()
        print(df['Prediction'])
        #df['predicted_column'].replace(-1, 0)
        #return df.to_json()
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(BASE_DIR, "prediction_output")

        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "output.csv")

        df.to_csv(output_path, index=False)
        table_html = df.to_html(classes='table table-striped', index=False)

        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "table": table_html,
                "total_records": total_records,
                "phishing_count": phishing_count,
                "legitimate_count": legitimate_count
            }
        )
        
    except Exception as e:
            raise NetworkSecurityException(e,sys)

    
if __name__=="__main__":
    app_run(app,host="0.0.0.0",port=8000)
