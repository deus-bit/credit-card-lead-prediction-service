from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class CreditLeadInput(BaseModel):
    age: int = Field(description="Edad del cliente", examples=[45])
    gender: Literal["Male", "Female"] = Field(
        description="Género: 'Male' o 'Female'", examples=["Male"]
    )
    occupation: Literal["Salaried", "Self_Employed", "Other"] = Field(
        description="Ocupación: 'Salaried', 'Self_Employed', 'Other'",
        examples=["Salaried"],
    )
    vintage: int = Field(description="Antigüedad en meses", examples=[24])
    credit_product: Literal["Yes", "No", "Unknown"] = Field(
        description="Tiene producto de crédito: 'Yes', 'No', 'Unknown'",
        examples=["No"],
    )
    avg_account_balance: float = Field(
        description="Balance promedio en cuenta", examples=[150000.0]
    )
    is_active: Literal["Yes", "No"] = Field(
        description="Cliente activo: 'Yes' o 'No'", examples=["Yes"]
    )


class Prediction(BaseModel):
    label: str
    probability: float


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


def preprocess_input(data: CreditLeadInput, scaler) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "Gender": int(data.gender == "Male"),
                "Age": data.age,
                "Vintage": data.vintage,
                "Credit_Product": int(data.credit_product == "Yes"),
                "Avg_Account_Balance": data.avg_account_balance,
                "Is_Active": int(data.is_active == "Yes"),
                "Occupation_Other": int(data.occupation == "Other"),
                "Occupation_Salaried": int(data.occupation == "Salaried"),
                "Occupation_Self_Employed": int(data.occupation == "Self_Employed"),
            }
        ]
    )

    # Aplicar el scaler cargado
    df = scaler.transform(df)

    # # Reordenar columnas para que coincidan exactamente con el entrenamiento
    # df = df.reindex(columns=feature_columns, fill_value=0)

    return df


@app.post("/predict")
async def predict_lead(lead: CreditLeadInput) -> Prediction:
    if not ml_models["model"]:
        raise HTTPException(status_code=500, detail="El modelo no está cargado.")

    try:
        model = ml_models["model"]
        scaler = ml_models["scaler"]
        # columns = ml_models["columns"]

        processed_data = preprocess_input(lead, scaler)

        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        label = "Interesado" if prediction == 1 else "No Interesado"

        return Prediction(label=label, probability=round(float(probability), 4))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error en la predicción: {str(e)}")


@app.get("/")
def read_root():
    return {"status": "API Online", "model_loaded": ml_models["model"] is not None}
