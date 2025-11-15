import os, joblib
import argparse
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
import uvicorn
import time
import json

app = FastAPI()

def convert_to_str(x):
    return x.astype(str)
    
# Cargar modelo al iniciar el servidor
model_path = "/app/model/last_model.joblib"
model = joblib.load(model_path)
    
@app.post("/predict")
def predict(payload: dict = Body(...)):
    try:
        start_time = time.time()
        
        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            raise HTTPException(status_code=400, detail="Body debe incluir 'data' (lista no vacía).")
        
        #Creamos un DataFrame
        df = pd.DataFrame(data)
        preds = model.predict(df)
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        with open("/app/model/metrics.json", 'r') as file:
            data = json.load(file)
            data['avg_response_time_seconds'] = elapsed_time
            data['predictions_served'] += len(preds)

        with open("/app/model/metrics.json", mode="w", encoding="utf-8") as write_file:
            json.dump(data, write_file)
        return {"predictions": preds.tolist()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

@app.post("/metrics")
def metrics():
    try:
        with open("/app/model/metrics.json", 'r') as file:
            data = json.load(file)
        return data
    except HTTPException:
        raise

def batch_mode():
    start_time = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Ruta al .parquet de entrada")
    parser.add_argument("--output", help="Ruta al .parquet de salida")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    y_pred = model.predict(df)
    
    out = pd.DataFrame({"prediction": y_pred})
    out.to_parquet(args.output, index=False)
    
    end_time = time.time()
    elapsed_time = end_time - start_time

    with open('/app/model/metrics.json', 'r') as file:
        data = json.load(file)
    
    data['avg_response_time_seconds'] = elapsed_time
    data['predictions_served'] = data['predictions_served'] + len(y_pred)

    with open("/app/model/metrics.json", mode="w", encoding="utf-8") as write_file:
        json.dump(data, write_file)

if __name__ == "__main__":
    # Detectar si se pasó argumento --input → modo batch
    if any(arg.startswith("--input") for arg in os.sys.argv):
        batch_mode()
    else:
        uvicorn.run("app:app", host="0.0.0.0", port=8000)