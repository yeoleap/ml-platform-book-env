"""Upload the two files the chapter 5 pipeline reads from storage.

1. upstream/chapter-5/scripts/feast/feature_store.yaml -> feature-registry/feature_store_k8s.yaml
   (cluster addresses; the chapter 4 file in the same bucket points at localhost)
2. upstream/chapter-4/feast/data/entity_df.parquet -> feature-data-sets/entity_df.parquet
   (the users and timestamps to predict)

    python book/ch05/upload_inputs.py
"""

from pathlib import Path

from minio import Minio

ROOT = Path(__file__).resolve().parents[2]
client = Minio("localhost:9000", access_key="minio", secret_key="minio123", secure=False)
uploads = [
    ("feature-registry", "feature_store_k8s.yaml",
     ROOT / "upstream" / "chapter-5" / "scripts" / "feast" / "feature_store.yaml"),
    ("feature-data-sets", "entity_df.parquet",
     ROOT / "upstream" / "chapter-4" / "feast" / "data" / "entity_df.parquet"),
]
for bucket, name, path in uploads:
    if not client.bucket_exists(bucket):
        raise SystemExit(f"Bucket '{bucket}' does not exist. Create it first.")
    client.fput_object(bucket, name, str(path))
    print(f"uploaded {path.relative_to(ROOT)} -> s3://{bucket}/{name}")
for bucket in ("feature-registry", "feature-data-sets"):
    print(bucket, sorted(o.object_name for o in client.list_objects(bucket)))
