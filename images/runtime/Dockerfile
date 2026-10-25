FROM python:3.11-slim-bookworm
WORKDIR /opt
COPY examples/requirements.txt /opt/requirements.txt
RUN pip install --no-cache-dir -r /opt/requirements.txt
COPY examples/feature_repo /opt/feature_repo
# Cluster addresses for KFP task Pods. S3 keys come from a Kubernetes secret.
ENV AWS_DEFAULT_REGION=us-east-1 \
    AWS_ENDPOINT_URL=http://minio-service.kubeflow.svc.cluster.local:9000 \
    FEAST_S3_ENDPOINT_URL=http://minio-service.kubeflow.svc.cluster.local:9000 \
    MLFLOW_S3_ENDPOINT_URL=http://minio-service.kubeflow.svc.cluster.local:9000 \
    MLFLOW_TRACKING_URI=http://mlflow-service.mlflow.svc.cluster.local:5000 \
    REDIS_CONNECTION_STRING=redis-deployment-master.redis.svc.cluster.local:6379
