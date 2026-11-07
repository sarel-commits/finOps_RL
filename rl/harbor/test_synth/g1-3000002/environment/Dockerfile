FROM python:3.12-slim
RUN pip install --no-cache-dir pandas==2.2.3 numpy==2.1.3
WORKDIR /app
COPY anomaly.json data.csv /app/
