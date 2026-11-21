import boto3
s3 = boto3.client("s3", endpoint_url="http://127.0.0.1:9000",
                  aws_access_key_id="minio", aws_secret_access_key="minio123")
names = [b["Name"] for b in s3.list_buckets()["Buckets"]]
print("before:", names)
if "ch08-datasets" not in names:
    s3.create_bucket(Bucket="ch08-datasets")
s3.upload_file("DATASET.tar.gz", "ch08-datasets", "DATASET.tar.gz")
print([(o["Key"], o["Size"]) for o in s3.list_objects_v2(Bucket="ch08-datasets")["Contents"]])
