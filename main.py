from contextlib import asynccontextmanager

import joblib
import pandas as pd

# import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class CreditLeadInput(BaseModel):
    age: int = Field(description="Edad del cliente", examples=[45])
    gender: str = Field(description="Género: 'Male' o 'Female'", examples=["Male"])
    occupation: str = Field(
        description="Ocupación: 'Salaried', 'Self_Employed', 'Entrepreneur', 'Other'",
        examples=["Salaried"],
    )
    vintage: int = Field(description="Antigüedad en meses", examples=[24])
    credit_product: str = Field(
        description="Tiene producto de crédito: 'Yes', 'No', 'Unknown'",
        examples=["No"],
    )
    avg_account_balance: float = Field(
        description="Balance promedio en cuenta", examples=[150000.0]
    )
    is_active: str = Field(description="Cliente activo: 'Yes' o 'No'", examples=["Yes"])


# Objetos importados (exportados desde el notebook)
ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Cargar modelo, escalador y columnas al iniciar la app
    try:
        ml_models["model"] = joblib.load("models/rf_model.joblib")
        ml_models["scaler"] = joblib.load("models/scaler.joblib")
        ml_models["columns"] = joblib.load("models/model_columns.joblib")
        print("Modelo y artefactos cargados correctamente.")
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        ml_models["model"] = None
    yield

    ml_models.clear()


app = FastAPI(title="Credit Card Lead Prediction API", lifespan=lifespan)


def preprocess_input(data: CreditLeadInput, feature_columns, scaler):
    """
    Transforma el JSON de entrada en un DataFrame con el formato exacto
    que espera el modelo (Scaling + OneHotEncoding).
    """
    # Crear DataFrame base
    df = pd.DataFrame([data.model_dump()])

    # Mapeos Manuales
    df["Gender"] = df["gender"].map({"Male": 1, "Female": 0})
    df["Is_Active"] = df["is_active"].map({"Yes": 1, "No": 0})
    df["Credit_Product"] = df["credit_product"].apply(lambda x: 1 if x == "Yes" else 0)

    # One-Hot Encoding (Occupation) Manual
    occupation_val = f"Occupation_{data.occupation}"

    # Inicializamos todas las columnas de One-Hot en 0
    for col in feature_columns:
        if col.startswith("Occupation_"):
            df[col] = 0

    # Si la ocupación actual existe en las columnas del modelo, ponemos un 1
    if occupation_val in feature_columns:
        df[occupation_val] = 1

    # Renombrar columnas para que coincidan con el scaler (Mayúsculas vs minúsculas del input)
    df = df.rename(
        columns={
            "age": "Age",
            "vintage": "Vintage",
            "avg_account_balance": "Avg_Account_Balance",
        }
    )

    # Aplicar el scaler cargado
    df = scaler.transform(df)

    # Reordenar columnas para que coincidan exactamente con el entrenamiento
    df = df.reindex(columns=feature_columns, fill_value=0)

    return df


@app.post("/predict")
async def predict_lead(lead: CreditLeadInput):
    if not ml_models["model"]:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")

    try:
        model = ml_models["model"]
        scaler = ml_models["scaler"]
        columns = ml_models["columns"]

        processed_data = preprocess_input(lead, columns, scaler)

        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        result = "Interesado" if prediction == 1 else "No Interesado"

        return {
            "prediction": int(prediction),
            "label": result,
            "probability_score": round(float(probability), 4),
            "input_summary": {"age": lead.age, "occupation": lead.occupation},
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "API Online", "model_loaded": ml_models["model"] is not None}
