"""Step 4. Batch inference on Kubeflow Pipelines inside the cluster.

Each task runs in its own Pod with the cluster addresses baked into the
runtime image; S3 keys come from the Kubernetes secret, not from parameters.
"""

import io
import time

import boto3
import pandas as pd
from kfp import Client, compiler, dsl, kubernetes
from kfp.dsl import Dataset, Input, Output

import common

IMAGE = "book-env/runtime:0.1"


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def read_data(source_uri: str, entities: Output[Dataset]):
    import pandas as pd

    frame = pd.read_parquet(source_uri)
    keys = ["user_id", "event_timestamp"]
    if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError("Entity input must have unique, non-empty ID and time")
    frame[keys].to_parquet(entities.path, index=False)


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def retrieve_features(feature_service: str, entities: Input[Dataset],
                      features: Output[Dataset]):
    import pandas as pd
    from feast import FeatureStore

    requests = pd.read_parquet(entities.path)
    store = FeatureStore(repo_path="/opt/feature_repo")
    frame = store.get_historical_features(
        entity_df=requests, features=store.get_feature_service(feature_service),
    ).to_df()
    aligned = requests.merge(frame, on=["user_id", "event_timestamp"],
                             how="left", validate="one_to_one")
    if aligned[["hours_per_week", "workclass"]].isna().any().any():
        raise ValueError("Historical features are missing for requested entities")
    aligned.to_parquet(features.path, index=False)


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def run_inference(model_uri: str, feature_service: str, features: Input[Dataset],
                  predictions: Output[Dataset]):
    import re
    import mlflow.sklearn
    import pandas as pd
    from mlflow import MlflowClient

    match = re.fullmatch(r"models:/([^/]+)/([0-9]+)", model_uri)
    if not match:
        raise ValueError("Use a fixed numeric registered-model version")
    version = MlflowClient().get_model_version(*match.groups())
    if version.tags.get("feast_feature_service") != feature_service:
        raise ValueError("Model was not trained with this feature service")
    model = mlflow.sklearn.load_model(model_uri)
    frame = pd.read_parquet(features.path)
    positive = list(model.classes_).index(1)
    frame["probability_gt_50k"] = model.predict_proba(
        frame[["hours_per_week", "workclass"]])[:, positive]
    frame["model_uri"] = model_uri
    frame["feature_service"] = feature_service
    frame.to_parquet(predictions.path, index=False)


@dsl.component(base_image=IMAGE, install_kfp_package=False)
def write_data(predictions: Input[Dataset], target_uri: str):
    import pandas as pd

    frame = pd.read_parquet(predictions.path)
    if not frame["probability_gt_50k"].between(0, 1).all():
        raise ValueError("Invalid prediction probability")
    frame.to_parquet(target_uri, index=False)
    print(frame.to_string(index=False))


@dsl.pipeline(name="income-book-env-batch")
def batch_pipeline(source_uri: str, model_uri: str, feature_service: str, target_uri: str):
    read = read_data(source_uri=source_uri)
    features = retrieve_features(feature_service=feature_service,
                                 entities=read.outputs["entities"])
    predict = run_inference(model_uri=model_uri, feature_service=feature_service,
                            features=features.outputs["features"])
    write = write_data(predictions=predict.outputs["predictions"], target_uri=target_uri)
    for task in (read, features, predict, write):
        task.set_caching_options(False)
        kubernetes.use_secret_as_env(
            task, secret_name="mlpipeline-minio-artifact",
            secret_key_to_env={"accesskey": "AWS_ACCESS_KEY_ID",
                               "secretkey": "AWS_SECRET_ACCESS_KEY"},
        )


if __name__ == "__main__":
    package = "income-book-env-batch.yaml"
    compiler.Compiler().compile(batch_pipeline, package)
    model_uri = open("selected_model_uri.txt", encoding="utf-8").read().strip()
    target_key = f"predictions-{int(time.time())}.parquet"
    client = Client(host=common.os.environ["KFP_HOST"])
    run = client.create_run_from_pipeline_package(
        package, run_name="income-book-env-batch", experiment_name="book-env",
        arguments={
            "source_uri": f"s3://{common.DATA_BUCKET}/requests.parquet",
            "model_uri": model_uri,
            "feature_service": common.FEATURE_SERVICE,
            "target_uri": f"s3://{common.RESULT_BUCKET}/{target_key}",
        },
    )
    print("KFP run:", f"{common.os.environ['KFP_HOST']}/#/runs/details/{run.run_id}")
    result = client.wait_for_run_completion(run.run_id, timeout=1200, sleep_duration=5)
    print("Run state:", result.state)
    if result.state != "SUCCEEDED":
        raise SystemExit("Pipeline did not succeed")
    body = boto3.client("s3").get_object(Bucket=common.RESULT_BUCKET, Key=target_key)["Body"].read()
    print(pd.read_parquet(io.BytesIO(body)).to_string(index=False))
