
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import joblib
from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from flask import Flask, render_template, request
import requests
from model_pipeline import prepare_data, train_model, evaluate_model, save_model, load_model

app = FastAPI()

# Define the input data model
class CustomerFeatures(BaseModel):
    account_length: float
    international_plan: int
    voice_mail_plan: int
    number_vmail_messages: float
    total_day_minutes: float
    total_day_calls: float
    total_day_charge: float
    total_eve_minutes: float
    total_eve_calls: float
    total_night_minutes: float
    total_night_calls: float
    total_intl_minutes: float
    total_intl_calls: float
    customer_service_calls: float

# Load the model
MODEL_PATH = "model.pkl"
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")

@app.post("/predict/")
async def predict(customer: CustomerFeatures):
    try:
        # Convert input data to numpy array in the correct order
        input_data = np.array([
            customer.account_length,
            customer.international_plan,
            customer.voice_mail_plan,
            customer.number_vmail_messages,
            customer.total_day_minutes,
            customer.total_day_calls,
            customer.total_day_charge,
            customer.total_eve_minutes,
            customer.total_eve_calls,
            customer.total_night_minutes,
            customer.total_night_calls,
            customer.total_intl_minutes,
            customer.total_intl_calls,
            customer.customer_service_calls
        ]).reshape(1, -1)
       
        # Make prediction
        prediction = model.predict(input_data)
       
        return {"prediction": int(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Churn Prediction API - Use POST /predict with customer data"}


class ModelParams(BaseModel):
    """Modèle de validation des hyperparamètres"""
    criterion: str = 'gini'
    splitter: str = 'best'
    max_depth: int = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    random_state: int = 42

@app.post("/retrain/")
def retrain(params: ModelParams):
    try:
        print("\n=== Début du réentraînement ===")
        print(f"Paramètres reçus : {params.dict()}")

        # Préparation des données
        X_train, X_test, y_train, y_test, scaler, pca_transformer = prepare_data()

        # Ajustement des paramètres
        adjusted_params = params.dict()

        # Création et entraînement du modèle avec les paramètres fournis
        print("\nCréation du nouveau modèle...")
        model = DecisionTreeClassifier(**adjusted_params)
        model.fit(X_train, y_train)

        # Sauvegarde du modèle et des objets de prétraitement
        save_model(model, scaler, pca_transformer)

        # Évaluation et retour des performances
        performance = evaluate_model(model, X_test, y_test)

        return {"message": "Modèle réentrainé avec succès!", "performance": performance}

    except Exception as e:
        return {"error": str(e)}
