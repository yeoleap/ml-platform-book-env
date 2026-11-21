from kfp import dsl
from kfp.dsl import Artifact, Input, Metrics, Output


@dsl.component(base_image="python:3.11", packages_to_install=["requests"])
def download_movielens(output_path_one: Output[Artifact], dataset: str = "ml-latest-small"):
    """Author's download with the dataset name as a parameter and TLS verification on."""
    import hashlib

    import requests

    url = f"https://files.grouplens.org/datasets/movielens/{dataset}.zip"
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    digest = hashlib.sha256()
    with open(output_path_one.path, "wb") as file:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                file.write(chunk)
                digest.update(chunk)
    output_path_one.metadata["source_url"] = url
    output_path_one.metadata["sha256"] = digest.hexdigest()
    print(url, digest.hexdigest())


@dsl.component(base_image="python:3.11")
def unzip_movielens(input_path: Input[Artifact], ratings_output_path: Output[Artifact],
                    movies_output_path: Output[Artifact], dataset: str = "ml-latest-small"):
    import zipfile

    with zipfile.ZipFile(input_path.path, "r") as z:
        with open(ratings_output_path.path, "wb") as f:
            f.write(z.read(f"{dataset}/ratings.csv"))
        with open(movies_output_path.path, "wb") as f:
            f.write(z.read(f"{dataset}/movies.csv"))


@dsl.component(base_image="python:3.11", packages_to_install=["scikit-learn", "pandas", "fastparquet"])
def split_ratings(input_parquet: Input[Artifact], dataset_path: Output[Artifact], random_state: int = 42):
    """Author's split_dataset without the two user_id/item_id lines."""
    import os

    import pandas as pd
    from sklearn.model_selection import train_test_split

    ratings_df = pd.read_parquet(input_parquet.path)
    print("columns:", list(ratings_df.columns), "rows:", len(ratings_df))
    train, test = train_test_split(ratings_df, test_size=0.25, random_state=random_state)
    val, test = train_test_split(test, test_size=0.10 / 0.25, random_state=random_state)
    os.mkdir(dataset_path.path)
    for name, df in [("train", train), ("val", val), ("test", test)]:
        df.to_parquet(os.path.join(dataset_path.path, f"{name}.parquet.gzip"), compression="gzip")
        print(name, df.shape)


@dsl.component(base_image="python:3.11", packages_to_install=["boto3"])
def put_to_s3(inputFile: Input[Artifact], upload_file_name: str = "", bucket: str = "datasets",
              prefix: str = "ml-latest-small"):
    """Author's put_to_minio with a prefix parameter and an idempotent bucket check."""
    import os

    import boto3
    from botocore.exceptions import ClientError

    s3 = boto3.client("s3", endpoint_url="http://minio-service.kubeflow:9000",
                      aws_access_key_id="minio", aws_secret_access_key="minio123")
    try:
        s3.head_bucket(Bucket=bucket)
    except ClientError:
        s3.create_bucket(Bucket=bucket)
    if os.path.isdir(inputFile.path):
        files = [(os.path.join(inputFile.path, f), f) for f in sorted(os.listdir(inputFile.path))]
    else:
        files = [(inputFile.path, upload_file_name or os.path.basename(inputFile.path))]
    for local, name in files:
        s3.upload_file(local, bucket, f"{prefix}/{name}")
        print(f"uploaded s3://{bucket}/{prefix}/{name} ({os.path.getsize(local)} bytes)")


@dsl.component(base_image="python:3.11", packages_to_install=["pyarrow", "pandas"])
def qa_movielens(report: Output[Metrics], bucket: str = "datasets", dataset: str = "ml-latest-small",
                 min_total_rows: int = 100000):
    """Checks all splits and movies, not only the train column count."""
    import pandas as pd
    from pyarrow import fs, parquet

    s3 = fs.S3FileSystem(endpoint_override="http://minio-service.kubeflow:9000",
                         access_key="minio", secret_key="minio123", scheme="http")

    def read(name):
        with s3.open_input_file(f"{bucket}/{dataset}/{name}.parquet.gzip") as f:
            return parquet.read_table(f).to_pandas()

    errors = []
    splits = {name: read(name) for name in ["train", "val", "test"]}
    movies = read("movies")
    total = sum(len(df) for df in splits.values())
    expected = ["userId", "movieId", "rating", "timestamp"]
    for name, df in splits.items():
        if list(df.columns) != expected:
            errors.append(f"{name} columns {list(df.columns)}")
            continue
        nulls = int(df[expected].isna().sum().sum())
        out_of_range = int((~df.rating.between(0.5, 5.0)).sum())
        print(f"{name}: rows={len(df)} share={len(df) / total:.3f} nulls={nulls} rating_out_of_range={out_of_range}")
        report.log_metric(f"{name}_rows", len(df))
        if nulls or out_of_range:
            errors.append(f"{name} nulls={nulls} out_of_range={out_of_range}")
    keys = {n: set(zip(df.userId, df.movieId)) for n, df in splits.items()}
    overlap = len(keys["train"] & keys["val"]) + len(keys["train"] & keys["test"]) + len(keys["val"] & keys["test"])
    all_ratings = pd.concat(splits.values())
    unknown_movies = int((~all_ratings.movieId.isin(movies.movieId)).sum())
    print(f"total={total} duplicate_keys_across_splits={overlap} ratings_with_unknown_movie={unknown_movies}")
    print("timestamp range:", pd.to_datetime(all_ratings.timestamp.min(), unit="s"),
          pd.to_datetime(all_ratings.timestamp.max(), unit="s"))
    report.log_metric("total_rows", total)
    report.log_metric("duplicate_keys_across_splits", overlap)
    report.log_metric("unknown_movies", unknown_movies)
    if total < min_total_rows:
        errors.append(f"total rows {total} < {min_total_rows}")
    if overlap or unknown_movies:
        errors.append(f"overlap={overlap} unknown_movies={unknown_movies}")
    if errors:
        raise SystemExit("QA failed: " + "; ".join(errors))
    print("QA passed!")
