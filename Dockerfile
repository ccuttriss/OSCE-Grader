# Dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV OSCE_DATA_DIR=/data
ENV OSCE_SERVER_MODE=1
ENV OSCE_LOG_JSON=1
VOLUME ["/data"]

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
