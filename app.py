# -*- coding: utf-8 -*-
"""
Created on Sat May  3 16:49:11 2025

@author: abnkumar
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

# Initialize FastAPI
app = FastAPI()

# Load model and encoders
model = joblib.load('saved_model.pkl')
label_encoders = joblib.load('label_encoders.pkl')
mlb = joblib.load('multi_label_binarizer.pkl')

# Define expected columns
columns = ['IP', 'IR', 'Market Code', 'Issue Currency', 'Issue Price', 'Face Value']
categorical_columns = ['IP', 'IR', 'Market Code', 'Issue Currency']

# Define input schema
class PredictionInput(BaseModel):
    features: list  # a list of values

@app.post("/predict")
def predict(input_data: PredictionInput):
    # Convert input list to DataFrame
    new_data = pd.DataFrame([input_data.features], columns=columns)
    
    # Encode categorical columns
    for col in categorical_columns:
        encoder = label_encoders[col]
        new_data[col] = encoder.transform(new_data[col])

    # Make prediction
    y_pred = model.predict(new_data)

    # Decode prediction
    predicted_labels = mlb.inverse_transform(y_pred)

    # Return the prediction
    return {"predicted_important_fields": predicted_labels}
