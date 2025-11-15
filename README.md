# Despliegue Completo de un Modelo de Machine Learning en Docker

### Requisitos
- Docker Desktop instalado
- WSL 2 instalado

### Instalacion

El repositorio se debe clonar o descargar como zip.

1. Clonar el repositorio:

Se puede realizar como una descarga de zip desde el siguiente link [Repositorio Lab1 PD](https://github.com/Jose-A-P/Lab2-ProductDevelopment.git)

o utilizando la terminal desde la ubicacion deseada:
```console
git clone https://github.com/Jose-A-P/Lab2-ProductDevelopment.git
```
## Como entrenar el modelo

Para poder entrenar el modelo debemos de ubicarnos desde WSL en el directorio donde se encuentra clonado o descargado el repositorio.

Al haber montado la carpeta, se debe de ejecutar la siguiente instruccion para crear el contendero 'model-train':
```console
docker build -t model-train:latest ./train
```

Este creara el contenedor, instalando la imagen y librerias necesarias. Luego de tenerlo creado, podemos realizar el entrenamiento del modelo con:
```console
docker run --rm \
  -v "$(pwd)/model:/output" \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/train/config.yaml:/app/config.yaml" \
  model-train:latest
  ```

  El entrenamiento del modelo es configurable desde el archivo 'config.yaml' ubicado en la carpeta train. Este nos permite modificar tanto los parametros del optimizador (utilizado Optuna en esta version) y el espacio de parametros posibles para el modelo.
  - `n_trials`: es la cantidad de iteraciones que realizara optuna en su estudio.
  - `direction`: dependiendo de la metrica utilizada se puede ajustar como maximizar o minimizar el resultado. Si se usa accuracy, este debe maximizar el estudio.
  - `metric`: es la metrica que utilizara optuna para el estudio.
  - `n_estimators`: indica la cantidad de 'arboles' que tendra el modelo.
  - `max_depth`: indica la profundida que tendra cada arbol creado.

## Cómo predecir con el modelo entrenado (modo batch o usando archivo)
  Para poder predecir con el modelo entrenado debemos de ubicarnos desde WSL en el directorio donde se encuentra clonado o descargado el repositorio.

Al haber montado la carpeta, se debe de ejecutar la siguiente instruccion para crear el contendero 'model-serve' (si no se ha montado previamente):
```console
docker build -t model-serve:latest ./serve
```

Para poder predecir con este modelo utilizamos el siguiente comando:
```console
docker run --rm \
-v "$(pwd)/data:/data" \
-v "$(pwd)/model:/app/model" \
model-serve:latest \
--input /data/input_test.parquet \
--output /data/output_preds.parquet
```

Se utiliza el archivo `input_test.parquet` de la carpeta `data` para generar predicciones con el modelo, siendo estas almacenadas en el archivo `output_preds.parquet` de la carpeta `data`.

Los dos archivos que encontramos en data son:
- `input_test.parquet`: Este contiene una muestra de las variables que puede utilizar el modelo para predecir, se puede generar como una muestra aleatoria del data original extraido de `data\hotel_bookings.csv`.
- `output_preds.parquet`: Contiende las predicciones de la muestra de datos ejecutada.

## Cómo predecir con el modelo entrenado (modo API)
  Para poder predecir con el modelo entrenado debemos de ubicarnos desde WSL en el directorio donde se encuentra clonado o descargado el repositorio.

Al haber montado la carpeta, se debe de ejecutar la siguiente instruccion para crear el contendero 'model-serve' (si no se ha montado previamente):
```console
docker build -t model-serve:latest ./serve
```

Ahora, para poner el servidor en marcha se utiliza el siguiente comando:
```console
docker run --rm -p 8000:8000 \
-v "$(pwd)/model:/app/model" \
model-serve:latest
```

El modo API cuenta con dos endpoints, los cuales son `/predict` y `/metrics`

### ---/predict en modo API---
Permite enviar un payload con los datos de cada observacion individual que se busque predecir. Estas deben de incluir el nombre de la columna y su valor.

Para probar este modo se puede utilizar el notebook `test_predict_api.ipynb` del repositorio. Este prepara el payload utilizando una muestra aleatoria de `hotel_bookings.csv`. Al ejecutarlo el servidor devuelve un json con las predicciones para cada uno de los puntos.

### ---/metrics en modo API---
Solicita al servidor las metricas del modelo, siendo:
- "model": Indica el modelo entrenado. Para esta version unicamente se regresa RandomForest ya que es el modelo de base.
- "accuracy": regresar la precision del ultimo modelo entrenado. Actualizado al entrenar el modelo con el contenedor `model-train`
- "predictions_served": Muestra la cantidad de predicciones generadas con el contenedor `model-serve`. Actualizado tanto en modo batch como API.
- "avg_response_time_ms": Muestra el tiempo requerido para generar las predicciones en el modo elegido. Actualizado tanto en modo batch como API.
- "last_training": regresar la fecha de entrenamiento del ultimo modelo. Actualizado al entrenar el modelo con el contenedor `model-train`
