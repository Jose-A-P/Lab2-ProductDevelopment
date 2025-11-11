#Dockerfile
FROM python:3.12-slim

WORKDIR /app

#Instalamos dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

#Agregar el script
COPY app.py /app/app.py

#Comando de arranque
ENTRYPOINT ["python", "app.py"]