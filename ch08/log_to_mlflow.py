"""Record one KFP training run in MLflow and register it with an explicit release decision."""
import csv, os, sys, yaml
import mlflow
from mlflow import MlflowClient

os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000")
os.environ.setdefault("AWS_ACCESS_KEY_ID", "minio")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minio123")
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ch08-id-card")

kfp_run_id = sys.argv[1]
art = "kfp_artifacts/model_output/train_run"
args = yaml.safe_load(open(f"{art}/args.yaml"))
# validate-model Metrics artifact of this run (val = used to pick best.pt, test = never seen)
kfp_metrics = {"map50": 0.995, "map50_95": 0.930354, "map75": 0.995,
               "test_map50": 0.995, "test_map50_95": 0.950992}
stress = {"rot90": 1, "blur8": 0, "small": 0, "dark": 1}               # stress_review.py result

with mlflow.start_run(run_name=f"kfp-{kfp_run_id[:8]}") as run:
    mlflow.set_tags({"kfp_run_id": kfp_run_id, "dataset_uri": "s3://ch08-datasets/DATASET.tar.gz",
                     "dataset_note": "synthetic ID-card-like images, clip-level split (no clip shared across splits)",
                     "author_code": "practical-mlops/object-detection-project@c6afb11",
                     "image": "ultralytics/ultralytics:8.3.40-arm64"})
    mlflow.log_params({k: args[k] for k in ["model", "epochs", "batch", "imgsz", "optimizer", "seed"]})
    for row in csv.DictReader(open(f"{art}/results.csv")):
        row = {k.strip(): v for k, v in row.items()}
        step = int(row["epoch"])
        mlflow.log_metrics({"train_box_loss": float(row["train/box_loss"]),
                            "val_box_loss": float(row["val/box_loss"]),
                            "val_map50_95_epoch": float(row["metrics/mAP50-95(B)"])}, step=step)
    mlflow.log_metrics({f"kfp_val_{k}" if not k.startswith("test") else f"kfp_{k}": v
                        for k, v in kfp_metrics.items()})
    mlflow.log_metric("stress_detect_rate", sum(stress.values()) / len(stress))
    mlflow.log_artifacts(art, artifact_path="train_run")
    mlflow.log_artifact("kfp_artifacts/data_yaml/data.yaml", artifact_path="config")
    run_id = run.info.run_id

c = MlflowClient()
name = "ch08-id-card-detector"
try:
    c.create_registered_model(name, description="Chapter 8 lab model (synthetic data, CPU, 5 epochs)")
except Exception:
    pass
mv = c.create_model_version(name, source=f"runs:/{run_id}/train_run/weights", run_id=run_id,
                            tags={"kfp_test_map50_95": "0.951", "manual_review": "fail: blur, small",
                                  "release_decision": "hold"})
print("mlflow run_id:", run_id)
print("registered:", name, "version", mv.version, "status", mv.status)
