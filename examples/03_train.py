"""Step 3. Train from Feast features and record the run in the MLflow server."""

import hashlib
import json
import os

import mlflow
import mlflow.data
import mlflow.sklearn
import pandas as pd
from feast import FeatureStore
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

import common

store = FeatureStore(repo_path=common.FEATURE_REPO)
service = store.get_feature_service(common.FEATURE_SERVICE)
labels_uri = f"s3://{common.DATA_BUCKET}/labels.parquet"
labels = pd.read_parquet(labels_uri)
data = store.get_historical_features(entity_df=labels, features=service).to_df()
columns = ["hours_per_week", "workclass"]
if len(data) != len(labels) or data[columns].isna().any().any():
    raise ValueError("Every label row needs point-in-time features")

definitions = os.path.join(common.FEATURE_REPO, "features.py")
lineage = {
    "feast_feature_service": common.FEATURE_SERVICE,
    "feast_definitions_sha256": hashlib.sha256(open(definitions, "rb").read()).hexdigest(),
}
train, valid = train_test_split(data, test_size=0.25, random_state=23, stratify=data["target"])
model = Pipeline([
    ("transform", ColumnTransformer([
        ("workclass", OneHotEncoder(handle_unknown="ignore"), ["workclass"]),
        ("hours", "passthrough", ["hours_per_week"]),
    ])),
    ("classifier", RandomForestClassifier(n_estimators=80, max_depth=5, random_state=23)),
])

mlflow.set_experiment("income-book-env")
with mlflow.start_run(run_name="forest") as run:
    model.fit(train[columns], train["target"])
    auc = roc_auc_score(valid["target"], model.predict_proba(valid[columns])[:, 1])
    mlflow.log_params({"max_depth": 5, "rows": len(data),
                       "feature_refs": json.dumps(["hours:hours_per_week", "hours:workclass"])})
    mlflow.set_tags(lineage)
    mlflow.log_input(mlflow.data.from_pandas(
        data.drop(columns=["event_timestamp"]), source=labels_uri,
        name="income-training-from-feast", targets="target"), context="training")
    mlflow.log_metric("validation_auc", auc)
    mlflow.log_artifact(definitions, artifact_path="feast")
    info = mlflow.sklearn.log_model(
        model, name="model",
        signature=infer_signature(train[columns], model.predict(train[columns])),
        input_example=train[columns].head(2),
    )

registered = mlflow.register_model(info.model_uri, common.MODEL_NAME)
client = MlflowClient()
for key, value in lineage.items():
    client.set_model_version_tag(common.MODEL_NAME, registered.version, key, value)
model_uri = f"models:/{common.MODEL_NAME}/{registered.version}"
reloaded = mlflow.sklearn.load_model(model_uri)
assert (reloaded.predict(valid[columns]) == model.predict(valid[columns])).all()
with open("selected_model_uri.txt", "w", encoding="utf-8") as stream:
    stream.write(model_uri)
print({"run_id": run.info.run_id, "model_uri": model_uri, "validation_auc": round(auc, 4)})
