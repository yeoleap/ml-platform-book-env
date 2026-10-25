"""Connection settings for running the examples from the laptop.

`python bookenv.py forward` exposes the cluster services on localhost. Values
already set in the environment take precedence, so the same scripts also run
inside the cluster with cluster addresses.
"""

import os

DEFAULTS = {
    "AWS_ACCESS_KEY_ID": "minio",
    "AWS_SECRET_ACCESS_KEY": "minio123",
    "AWS_DEFAULT_REGION": "us-east-1",
    "AWS_ENDPOINT_URL": "http://localhost:9000",
    "FEAST_S3_ENDPOINT_URL": "http://localhost:9000",
    "MLFLOW_S3_ENDPOINT_URL": "http://localhost:9000",
    "MLFLOW_TRACKING_URI": "http://localhost:5000",
    "REDIS_CONNECTION_STRING": "localhost:6379,password=redis123",
    "KFP_HOST": "http://localhost:8080",
}
for key, value in DEFAULTS.items():
    os.environ.setdefault(key, value)

FEATURE_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feature_repo")
FEATURE_SERVICE = "income_model_v1"
MODEL_NAME = "income-book-env"
DATA_BUCKET = "mlflow-datasets"
FEATURE_BUCKET = "feature-data-sets"
RESULT_BUCKET = "inference-datasets"
