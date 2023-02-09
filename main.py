from typing import Union

from fastapi import FastAPI

from pydantic import BaseModel
import pandas as pd
import joblib

X_test=pd.read_csv("X_test_sample.csv",index_col='SK_ID_CURR')
model_predict=joblib.load("regression.joblib")

class Predict_class(BaseModel):
    client_id: int

app = FastAPI()

@app.get("/")
def read_root():
    return {"Bienvenue dans notre application via PYTHON"}

@app.post("/predict_client")
def fct_predict(itemClient:Predict_class):
    client=itemClient.dict()
    client_id=client['client_id']
    try:
        client_data=X_test.loc[[client_id]]
        prediction=model_predict.predict(client_data)
        prediction_proba=model_predict.predict_proba(client_data)
        output_val={'prediction':int(prediction[0]),'probability':round(prediction_proba[0].max(),2)}
        return output_val
    except:
        return ("Client Nnt found")
