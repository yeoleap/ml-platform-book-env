"""Step 1. Create synthetic income data and upload it to S3 buckets."""

import io

import boto3
import numpy as np
import pandas as pd

import common

rng = np.random.default_rng(23)
users = np.arange(1000, 1600)
classes = ["private", "public", "self-employed"]
observed = pd.Timestamp("2025-04-01", tz="UTC") + pd.to_timedelta(
    rng.integers(0, 30, size=users.size), unit="D"
)
hours = rng.integers(20, 61, size=users.size).astype(float)
workclass = rng.choice(classes, size=users.size)
history = pd.concat([
    pd.DataFrame({
        "user_id": users, "event_timestamp": observed,
        "hours_per_week": hours, "workclass": workclass,
    }),
    pd.DataFrame({
        "user_id": users, "event_timestamp": observed + pd.Timedelta(days=45),
        "hours_per_week": rng.integers(20, 61, size=users.size).astype(float),
        "workclass": rng.choice(classes, size=users.size),
    }),
    pd.DataFrame({
        "user_id": [42, 42],
        "event_timestamp": pd.to_datetime(["2025-05-21T00:00:00Z", "2025-06-04T00:00:00Z"]),
        "hours_per_week": [36.0, 52.0],
        "workclass": ["private", "self-employed"],
    }),
], ignore_index=True)

effect = pd.Series(workclass).map({"private": 0.0, "public": -0.4, "self-employed": 0.8})
score = (hours - 40) / 8 + effect.to_numpy() + rng.normal(size=users.size)
labels = pd.DataFrame({
    "user_id": users,
    "event_timestamp": observed + pd.Timedelta(days=7),
    "target": (score > 0).astype(int),
})
requests = pd.DataFrame({
    "user_id": [42],
    "event_timestamp": pd.to_datetime(["2025-05-22T00:00:00Z"]),
})

s3 = boto3.client("s3")
existing = {b["Name"] for b in s3.list_buckets()["Buckets"]}
for bucket in (common.FEATURE_BUCKET, common.DATA_BUCKET, common.RESULT_BUCKET,
               "feature-registry", "mlflow-artifacts"):
    if bucket not in existing:
        s3.create_bucket(Bucket=bucket)
        print("created bucket", bucket)
for bucket, key, frame in [
    (common.FEATURE_BUCKET, "hours.parquet", history),
    (common.DATA_BUCKET, "labels.parquet", labels),
    (common.DATA_BUCKET, "requests.parquet", requests),
]:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    s3.put_object(Bucket=bucket, Key=key, Body=buffer.getvalue())
    print(f"s3://{bucket}/{key}: {len(frame)} rows")
