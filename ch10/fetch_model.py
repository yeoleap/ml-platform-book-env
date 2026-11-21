"""Copies best.pt of a registered model version from MLflow into detector/ and records where it came from.

usage (ch08 venv has mlflow): ../ch08/.venv/bin/python fetch_model.py [version]
"""
import hashlib
import json
import os
import shutil
import sys

import mlflow
from mlflow import MlflowClient

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minio123")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

name = "ch08-id-card-detector"
version = sys.argv[1] if len(sys.argv) > 1 else "1"
mv = MlflowClient().get_model_version(name, version)
print(f"{name} v{mv.version}: source={mv.source} run_id={mv.run_id}")
print("tags:", mv.tags)

local = mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{name}/{version}")
os.makedirs("detector", exist_ok=True)
dest = os.path.join("detector", "yolov8_custom.pt")
shutil.copy2(os.path.join(local, "best.pt"), dest)
sha256 = hashlib.sha256(open(dest, "rb").read()).hexdigest()

info = {"model_name": name, "model_version": mv.version, "run_id": mv.run_id, "source": mv.source,
        "sha256": sha256, "tags": mv.tags}
json.dump(info, open(os.path.join("detector", "model_info.json"), "w"), indent=2)
print(f"saved {dest} ({os.path.getsize(dest)} bytes) sha256={sha256[:12]}...")
