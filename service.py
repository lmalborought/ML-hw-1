from fastapi import FastAPI, File, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel
from typing import List
import pandas as pd
import numpy as np
from io import StringIO
import joblib

app = FastAPI()

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')

class Item(BaseModel):
    name: str
    year: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float
 
class Items(BaseModel):
    objects: List[Item]


def data_transform(df):
    df['mileage'] = df['mileage'].str.replace(r'\s*kmpl\s*|\s*km/kg\s*', '', regex=True).astype(float)
    df['engine'] = df['engine'].str.replace('CC', '').astype(int)
    df['max_power'] = df['max_power'].str.replace('bhp', '').astype(float)
    df['seats'] = df['seats'].astype(int)
    df = df.drop('torque', axis=1)
    df = df.select_dtypes(include=np.number)
    df = scaler.transform(df)
    return df

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.model_dump()])
    transform_data = data_transform(df)
    pred = model.predict(transform_data)
    return pred

@app.post("/predict_items")
async def predict_items(file: UploadFile):
    contents = await file.read()
    csv_data = StringIO(contents.decode('utf-8'))
    df = pd.read_csv(csv_data)

    df_transform = data_transform(df)
    pred = model.predict(df_transform)
    df['selling_price'] = pred

    stream = StringIO()
    df.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),media_type="csv")
    response.headers["Content-Disposition"] = "attachment; filename=result.csv"
    return response
