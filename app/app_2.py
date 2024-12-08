import numpy as np
import pandas as pd
import re
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import Optional, List
from joblib import load
from enum import IntEnum, Enum
from fastapi.encoders import jsonable_encoder
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif

model = load('C:\\py.projects\\app.py\\.venv\\model.joblib')
scaler = load('C:\\py.projects\\app.py\\.venv\\scaler.joblib')
ohe = load('C:\\py.projects\\app.py\\.venv\\ohe.joblib')
with open('C:\\py.projects\\app.py\\.venv\\feature_names.txt', 'r') as f:
    feature_names = f.read().splitlines()

app = FastAPI()

class Item(BaseModel):
    name: str
    year: int
    selling_price: int
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


def preprocess_item(item: Item) -> pd.DataFrame:
    data = pd.DataFrame([{
        "year": item.year,
        "km_driven": item.km_driven,
        "fuel": item.fuel,
        "seller_type": item.seller_type,
        "transmission": item.transmission,
        "owner": item.owner,
        "mileage": float(re.sub(r'[^\d\.]', '', item.mileage)),
        "engine": int(re.sub(r'[^\d]', '', item.engine)),
        "max_power": float(re.sub(r'[^\d\.]', '', item.max_power)),
        "seats": int(item.seats)
    }])
    # Разделяем на категориальные и числовые признаки
    cat_cols = ["fuel", "seller_type", "transmission", "owner"]
    num_cols = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]

    cat_transformed = pd.DataFrame(ohe.transform(data[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    num_transformed = pd.DataFrame(scaler.transform(data[num_cols]), columns=num_cols)

    # Объединяем числовые и категориальные данные
    processed_data = pd.concat([num_transformed, cat_transformed], axis=1)

    return processed_data

@app.get("/")
async def root():
    return {
        "Name": "Предсказание стоимости автомобиля",
        "Description": "Веб-сервис для применения построенной модели на новых данных"
    }

@app.post("/predict_item")
def predict_item(item: Item) -> float:
    processed_data = preprocess_item(item)
    prediction = model.predict(processed_data)
    return float(prediction[0])


@app.post("/predict_items")
def predict_items(items: Items) -> List[float]:
    data = pd.DataFrame([{
        "year": obj.year,
        "km_driven": obj.km_driven,
        "fuel": obj.fuel,
        "seller_type": obj.seller_type,
        "transmission": obj.transmission,
        "owner": obj.owner,
        "mileage": float(re.sub(r'[^\d\.]', '', obj.mileage)),
        "engine": int(re.sub(r'[^\d]', '', obj.engine)),
        "max_power": float(re.sub(r'[^\d\.]', '', obj.max_power)),
        "seats": int(obj.seats)
    } for obj in items.objects])

    cat_cols = ["fuel", "seller_type", "transmission", "owner"]
    num_cols = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]

    cat_transformed = pd.DataFrame(ohe.transform(data[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    num_transformed = pd.DataFrame(scaler.transform(data[num_cols]), columns=num_cols)
    processed_data = pd.concat([num_transformed, cat_transformed], axis=1)

    predictions = model.predict(processed_data)
    return predictions.tolist()


@app.post("/predict_file")
def predict_file(file: UploadFile) -> str:
    df = pd.read_csv(file.file)
    # Оставляем только нужные столбцы
    required_columns = ["year", "km_driven", "fuel", "seller_type", "transmission", "owner", "mileage", "engine",
                        "max_power", "seats"]
    df = df[required_columns]
    # Преобразуем числовые признаки
    df["mileage"] = df["mileage"].str.replace(r'[^\d\.]', '', regex=True).astype(float)
    df["engine"] = df["engine"].str.replace(r'[^\d]', '', regex=True).astype(int)
    df["max_power"] = df["max_power"].str.replace(r'[^\d\.]', '', regex=True).astype(float)
    # Разделяем на числовые и категориальные признаки
    cat_cols = ["fuel", "seller_type", "transmission", "owner"]
    num_cols = ["year", "km_driven", "mileage", "engine", "max_power", "seats"]
    # Кодируем и масштабируем
    cat_transformed = pd.DataFrame(ohe.transform(df[cat_cols]), columns=ohe.get_feature_names_out(cat_cols))
    num_transformed = pd.DataFrame(scaler.transform(df[num_cols]), columns=num_cols)
    processed_data = pd.concat([num_transformed, cat_transformed], axis=1)
    # Предсказание
    predictions = model.predict(processed_data)
    # Добавляем предсказания к исходным данным
    df["predicted_price"] = predictions
    result_file_path = "predictions.csv"
    df.to_csv(result_file_path, index=False)
    return f"Файл с предсказаниями сохранён: {result_file_path}"
