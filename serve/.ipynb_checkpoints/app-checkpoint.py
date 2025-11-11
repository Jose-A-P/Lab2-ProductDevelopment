import argparse
import pandas as pd 
import joblib

def convert_to_str(x):
    return x.astype(str)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/app/model/last_model.joblib", help = "Ruta a carpeta del modelo de MLFlow")
    parser.add_argument("--input", required=True, help = "Ruta al .parquet de entrada")
    parser.add_argument("--output", required=True, help = "Ruta al .parquet de salida")
    args = parser.parse_args()

    model = joblib.load(args.model)
    data = pd.read_parquet(args.input)
    
    y_pred = model.predict(data)

    out = pd.DataFrame({"prediction":y_pred})
    out.to_parquet(args.output, index=False)
    
if __name__=="__main__":
    main()