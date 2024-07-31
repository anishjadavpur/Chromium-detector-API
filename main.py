# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:09:25 2024

@author: ANISH SARKAR
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import json

app = FastAPI(
    title="Chromium Detector API",
    description="API for predicting the concentration of chromium samples."
    )

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    )

class ModelInput(BaseModel):
    B: float
    G: float
    R: float
    b: float
    H: float
    
model = joblib.load('best_chromium_regressor_pipeline2 (2).joblib')

# Endpoint to predict concentration
@app.post('/conc_prediction')
async def predict_concentration(input_parameters: ModelInput):
    # Extract input parameters from request body
    B = input_parameters.B
    G = input_parameters.G
    R = input_parameters.R
    b = input_parameters.b
    H = input_parameters.H
    
    # Create input list for prediction
    input_list = [B, G, R, b, H]
    
    # Make prediction using the loaded model
    prediction = model.predict([input_list])
    
    # Convert prediction to list for JSON serialization
    prediction_list = prediction.tolist()
    
    # Return prediction as JSON response
    return {"prediction": prediction_list}
