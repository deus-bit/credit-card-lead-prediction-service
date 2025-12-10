# 💳 Credit Card Lead Prediction API

## 🚀 Visión General del Proyecto

Este proyecto implementa una API utilizando **FastAPI** para predecir si un cliente bancario tiene una alta probabilidad de estar interesado en una tarjeta de crédito.

El modelo de predicción fue entrenado con el dataset **Credit Card Lead Prediction** de Kaggle, utilizando un **RandomForestClassifier** preprocesado para asegurar una alta precisión y una fácil integración en sistemas de terceros a través de un endpoint RESTful.

## 🛠️ Requisitos Previos

Necesitas tener instalado lo siguiente:

* **Python 3.13+**
* **pip** (Administrador de paquetes de Python)
* **Git** (Opcional, si clonas el repositorio)

## 📦 Configuración e Instalación Local

Sigue estos pasos para levantar el proyecto en tu máquina local.

### 1. Clonar el Repositorio (Opcional)

Si el proyecto está en un repositorio:

```bash
git clone <URL_DEL_REPOSITORIO>
cd credit_card_api

### 3. Configuración del Entorno Virtual (Instalación de Dependencias)
~~~powershell
python -m venv venv
.venv\Scripts\activate
pip install -r requirements.txt
~~~

### 4. Ejecución del Proyecto
~~~powershell
uvicorn main:app --reload
~~~

### 5. Ejemplo de Uso
~~~powershell
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "age": 40,
  "gender": "Male",
  "occupation": "Salaried",
  "vintage": 36,
  "credit_product": "No",
  "avg_account_balance": 1250000.50,
  "is_active": "Yes"
}'
~~~
