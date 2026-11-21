"""Uploads the synthetic ID-card archive to s3://ch07-raw through the forwarded S3 port."""
import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3", endpoint_url="http://127.0.0.1:9000",
                  aws_access_key_id="minio", aws_secret_access_key="minio123")
try:
    s3.head_bucket(Bucket="ch07-raw")
except ClientError:
    s3.create_bucket(Bucket="ch07-raw")
s3.upload_file("data/synthetic-idcards.tar.gz", "ch07-raw", "synthetic-idcards.tar.gz")
for obj in s3.list_objects_v2(Bucket="ch07-raw")["Contents"]:
    print(obj["Key"], obj["Size"])
