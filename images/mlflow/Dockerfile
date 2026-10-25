FROM python:3.11-slim-bookworm
COPY images/mlflow/requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
EXPOSE 5000
