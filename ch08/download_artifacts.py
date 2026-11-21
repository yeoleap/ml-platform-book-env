"""Download the train-model artifacts of one KFP run from the pipeline root (mlpipeline bucket)."""
import os, sys, boto3
run_id = sys.argv[1]
s3 = boto3.client("s3", endpoint_url="http://127.0.0.1:9000", aws_access_key_id="minio", aws_secret_access_key="minio123")
prefix = f"v2/artifacts/ch08-yolo-train-validate/{run_id}/train-model/"
n = 0
for page in s3.get_paginator("list_objects_v2").paginate(Bucket="mlpipeline", Prefix=prefix):
    for o in page.get("Contents", []):
        rel = o["Key"].split("/train-model/", 1)[1].split("/", 1)[1]   # drop execution id
        dest = os.path.join("kfp_artifacts", rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        s3.download_file("mlpipeline", o["Key"], dest)
        n += 1
print(f"downloaded {n} files to kfp_artifacts")
for root in ["kfp_artifacts/model_output", "kfp_artifacts/data_yaml"]:
    print(root, sorted(os.listdir(root)))
print(open("kfp_artifacts/data_yaml/data.yaml").read())
