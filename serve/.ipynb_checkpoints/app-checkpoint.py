import os, joblib
import argparse
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
import uvicorn

app = FastAPI()

def convert_to_str(x):
    return x.astype(str)
    
# Cargar modelo al iniciar el servidor
model_path = "/app/model/last_model.joblib"
model = joblib.load(model_path)

@app.post("/predict")
def predict(payload: dict = Body(...)):
    try:
        data = payload.get("data", [])
        if not isinstance(data, list) or not data:
            raise HTTPException(status_code=400, detail="Body debe incluir 'data' (lista no vacía).")
        
        #Creamos un DataFrame
        df = pd.DataFrame(data)
        preds = model.predict(df)
        return {"predictions": preds.tolist()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al predecir: {e}")

def batch_mode():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Ruta al .parquet de entrada")
    parser.add_argument("--output", help="Ruta al .parquet de salida")
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    y_pred = model.predict(df)
    out = pd.DataFrame({"prediction": y_pred})
    out.to_parquet(args.output, index=False)

if __name__ == "__main__":
    # Detectar si se pasó argumento --input → modo batch
    if any(arg.startswith("--input") for arg in os.sys.argv):
        batch_mode()
    else:
        uvicorn.run("app:app", host="0.0.0.0", port=8000)