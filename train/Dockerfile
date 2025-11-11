#Dockerfile
FROM python:3.12-slim

WORKDIR /app

#Instalamos dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

#Agregar el script
COPY train_model.py /app/train_model.py

#Comando de arranque
ENTRYPOINT ["python", "train_model.py"]