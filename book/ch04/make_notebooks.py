"""Build the chapter 4 notebooks from the book's code (practical-mlops/chapter-4).

Cells stay in the book's order. Cells added for this environment are marked
with [환경] in a markdown cell right above them.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


_ids = iter(range(1, 1000))


def md(text):
    return {"cell_type": "markdown", "id": f"cell-{next(_ids):03d}", "metadata": {}, "source": text.strip("\n")}


def code(text):
    return {"cell_type": "code", "id": f"cell-{next(_ids):03d}", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.strip("\n")}


def notebook(cells):
    return {
        "cells": cells,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 5,
    }


mlflow_cells = [
    md("""
# 4.1 MLflow로 실험 추적하기

책 4.1절과 공식 저장소 `chapter-4/mlflow/data_exploration_and_model_training.ipynb`의 셀을 순서대로 옮겼습니다.
`[환경]` 표시가 있는 셀만 이 실습 환경을 위해 추가했습니다.

실행 전 확인할 것
1. 다른 터미널에서 `python bookenv.py forward`가 실행 중이어야 합니다.
2. 저장소 UI에서 `mlflow-datasets` 버킷을 만들어 두어야 합니다.
3. 커널은 `.venv-book` 가상환경이어야 합니다.
"""),
    md("[환경] 데이터 경로: 공식 저장소를 `upstream/chapter-4`에 내려받은 경우의 위치입니다."),
    code("""
from pathlib import Path
DATA = Path("../../upstream/chapter-4/data/income_data.csv").resolve()
assert DATA.exists(), f"{DATA} 가 없습니다. SOP의 '책 코드 내려받기' 단계를 먼저 실행하세요."
print(DATA)
"""),
    md("## 1. MLflow가 아티팩트 저장소에 접근할 환경 변수 (책의 첫 셀)"),
    code("""
import os

# Set environment variables for mlflow to access the artifact store
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["AWS_ENDPOINT_URL"] = "http://localhost:9000"
"""),
    md("[환경] MLflow 3.0.0 클라이언트는 `MLFLOW_S3_ENDPOINT_URL`도 읽습니다. 같은 주소로 맞춥니다."),
    code("""
os.environ["MLFLOW_S3_ENDPOINT_URL"] = "http://localhost:9000"
"""),
    code("""
import pandas as pd
import os
import matplotlib.pyplot as plt
from plotnine import ggplot,geom_bar,theme_bw,labs,coord_flip,aes
import time
"""),
    md("## 2. 데이터 탐색 (책 4.1.1)"),
    code("""
df=pd.read_csv(DATA, index_col=False)
"""),
    code("""
print(df.head() ,'\\n')
print(df.info(), '\\n')
print(df.describe(), '\\n')
"""),
    code("""
if not os.path.exists("categorical_variable_plots"):
    os.makedirs("categorical_variable_plots")
for i in df.iloc[:,:-1].select_dtypes(include='object').columns:
    print(f'Variable {i}  \\n ')
    print(df[i].value_counts())
    plot = ggplot(df)+ geom_bar(aes(x=df[i], fill=df.Target), position='fill')+ theme_bw() + labs(title=f'Variable {i} ~ Target')+ coord_flip()
    plot.save(f"categorical_variable_plots/Variable {i}", verbose=False)
"""),
    md("## 3. MLflow 연결 (책 4.1.2)\n\n책은 `mlflow ui`로 PC에 추적 서버를 띄웁니다. 이 환경에서는 클러스터의 MLflow 서버가 같은 `localhost:5000`으로 연결되어 있습니다."),
    code("""
import mlflow
import uuid
"""),
    code("""
# setting tracking uri and experiment. If experiment name does not exist it will be created
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("income-classifier")
"""),
    code("""
with mlflow.start_run(run_name=f"eda-{uuid.uuid4()}"):
    for i in df.iloc[:,:-1].select_dtypes(include='object').columns:
        plot = ggplot(df)+ geom_bar(aes(x=df[i], fill=df.Target), position='fill')+ theme_bw() + labs(title=f'Variable {i} ~ Target')+ coord_flip()
        if not os.path.exists("categorical_variable_plots"):
            os.makedirs("categorical_variable_plots")
        plot.save(f"categorical_variable_plots/Variable {i}", verbose=False)
        mlflow.log_artifacts("categorical_variable_plots")
"""),
    md("## 4. 학습 데이터 분할"),
    code("""
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import pickle
target=df.Target
feature_df=df.drop('Target', axis=1)
#Codifying the predictors and the target
encoder=OneHotEncoder(sparse_output=False, drop='if_binary')
target=encoder.fit_transform(np.array(target).reshape(-1,1))
dummyfied_df=pd.get_dummies(feature_df, drop_first=True, sparse=False, dtype=float)
col_list = dummyfied_df.columns.to_list()
with open('column_list.pkl', 'wb') as f:
    pickle.dump(col_list, f)
X_train,X_test,y_train, y_test=train_test_split(dummyfied_df.reindex(columns=col_list,fill_value=0)
, target, train_size=0.80, shuffle=True)
"""),
    md("## 5. MinIO 버킷 확인과 업로드 함수"),
    code("""
from minio import Minio
minioClient = Minio("localhost:9000", access_key="minio", secret_key="minio123", secure=False)
found = minioClient.bucket_exists("mlflow-datasets")
if not found:
    print("PLEASE CREATE BUCKET IN MINIO BEFORE PROCEEDING")
else:
    print("mlflow-datasets bucket found")
"""),
    code("""
from io import BytesIO
def save_df_to_minio(df,bucket_name,path):
    csv_bytes = df.to_csv(index=False).encode('utf-8')
    csv_buffer = BytesIO(csv_bytes)

    minioClient.put_object(f'{bucket_name}',
                        f'{path}',
                            data=csv_buffer,
                            length=len(csv_bytes),
                            content_type='application/csv')
"""),
    md("## 6. Decision Tree (책 리스팅 4.5)"),
    code("""
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import  roc_auc_score
BUCKET_NAME = "mlflow-datasets"
with mlflow.start_run() as run:
    tree=DecisionTreeClassifier()
    run_id = run.info.run_id
    feature_df_path = f"income-classifier-datasets/feature_df-{run_id}.csv"
    save_df_to_minio(feature_df,BUCKET_NAME,feature_df_path)
    train_df = pd.concat([X_train,pd.Series(y_train.ravel())],axis=1)
    train_df_path = f"income-classifier-datasets/train-{run_id}.csv"
    save_df_to_minio(train_df,BUCKET_NAME,train_df_path)
    test_df = pd.concat([X_test,pd.Series(y_test.ravel())],axis=1)
    test_df_path = f"income-classifier-datasets/test-{run_id}.csv"
    save_df_to_minio(test_df,BUCKET_NAME,test_df_path)
    training_dataset = mlflow.data.from_pandas(train_df, source=f"{BUCKET_NAME}/{train_df_path}")
    test_dataset = mlflow.data.from_pandas(test_df, source=f"{BUCKET_NAME}/{test_df_path}")
    feature_dataset = mlflow.data.from_pandas(feature_df, source=f"{BUCKET_NAME}/{feature_df_path}")
    mlflow.log_input(training_dataset,context="training")
    mlflow.log_input(test_dataset,context="testing")
    mlflow.log_input(feature_dataset,context="reference")
    tree.fit(X_train, y_train.ravel())
    roc_auc_score_train = roc_auc_score(y_train==1, tree.predict_proba(X_train)[:,1])
    roc_auc_score_test = roc_auc_score(y_test==1, tree.predict_proba(X_test)[:,1])
    training_accuracy = tree.score(X_train, y_train)
    test_accuracy = tree.score(X_test, y_test)
    mlflow.log_artifact("column_list.pkl")
    mlflow.log_metric("roc_auc_score_train",roc_auc_score_train)
    print(f'Roc Auc Score train: {roc_auc_score_train}  \\n')
    mlflow.log_metric("roc_auc_score_test",roc_auc_score_test)
    print(f'Roc Auc Score test: {roc_auc_score_test}  \\n')
    mlflow.log_metric("training_accuracy",training_accuracy)
    print(f'Accuracy train : {training_accuracy}')
    mlflow.log_metric("test_accuracy",test_accuracy)
    print(f'Accuracy test : {test_accuracy}')
    mlflow.sklearn.log_model(tree,"income-classifier")
    mlflow.log_params(tree.get_params())
"""),
    md("## 7. Random Forest"),
    code("""
from sklearn.ensemble import RandomForestClassifier
with mlflow.start_run() as run:
    forest=RandomForestClassifier()
    start=time.time()
    run_id = run.info.run_id
    train_df = pd.concat([X_train,pd.Series(y_train.ravel())],axis=1)
    test_df = pd.concat([X_test,pd.Series(y_test.ravel())],axis=1)
    feature_df_path = f"income-classifier-datasets/feature_df-{run_id}.csv"
    train_df_path = f"income-classifier-datasets/train-{run_id}.csv"
    save_df_to_minio(train_df,BUCKET_NAME,train_df_path)
    save_df_to_minio(feature_df,BUCKET_NAME,feature_df_path)
    test_df_path = f"income-classifier-datasets/test-{run_id}.csv"
    save_df_to_minio(test_df,BUCKET_NAME,test_df_path)
    feature_dataset = mlflow.data.from_pandas(feature_df, source=f"{BUCKET_NAME}/{test_df_path}")
    training_dataset = mlflow.data.from_pandas(train_df, source=f"{BUCKET_NAME}/{train_df_path}")
    test_dataset = mlflow.data.from_pandas(test_df, source=f"{BUCKET_NAME}/{test_df_path}")
    mlflow.log_input(training_dataset,context="training")
    mlflow.log_input(test_dataset,context="testing")
    mlflow.log_input(feature_dataset,context="reference_features")
    forest.fit(X_train, y_train.ravel())
    end=time.time()
    roc_auc_score_train = roc_auc_score(y_train==1, forest.predict_proba(X_train)[:,1])
    roc_auc_score_test = roc_auc_score(y_test==1, forest.predict_proba(X_test)[:,1])
    training_accuracy = forest.score(X_train, y_train)
    test_accuracy = forest.score(X_test, y_test)
    mlflow.log_artifact("column_list.pkl")
    mlflow.log_metric("roc_auc_score_train",roc_auc_score_train)
    print(f'Roc Auc Score train: {roc_auc_score_train}  \\n')
    mlflow.log_metric("roc_auc_score_test",roc_auc_score_test)
    print(f'Roc Auc Score test: {roc_auc_score_test}  \\n')
    mlflow.log_metric("training_accuracy",training_accuracy)
    print(f'Accuracy train : {training_accuracy}')
    mlflow.log_metric("test_accuracy",test_accuracy)
    print(f'Accuracy test : {test_accuracy}')
    mlflow.sklearn.log_model(forest,"income-classifier")
    mlflow.log_params(forest.get_params())
"""),
    md("## 8. XGBoost와 자동 기록 (autolog)"),
    code("""
import xgboost as xgb
from sklearn.metrics import  accuracy_score
with mlflow.start_run() as run:
    mlflow.xgboost.autolog()
    n_round=30
    run_id = run.info.run_id
    dtrain= xgb.DMatrix(data=X_train, label=y_train.ravel())
    dtest= xgb.DMatrix(data=X_test, label=y_test.ravel())
    params={"objective":"binary:logistic",'colsample_bytree': 1,'learning_rate': 1,
                    'max_depth': 10 , 'subsample':1}
    model=xgb.train(params,dtrain, n_round)
    pred_train= model.predict(dtrain)
    pred_test=model.predict(dtest)
    model=xgb.train(params={"objective":"binary:hinge",'colsample_bytree': 1,'learning_rate': 1,
                    'max_depth': 10 , 'subsample':1}, dtrain=dtrain)
    pred_train= model.predict(dtrain)
    pred_test=model.predict(dtest)
    roc_auc_score_train = roc_auc_score(y_train==1, pred_train)
    roc_auc_score_test = roc_auc_score(y_test==1, pred_test)
    training_accuracy = accuracy_score(y_train, pred_train)
    test_accuracy = accuracy_score(y_test, pred_test)
    mlflow.log_metric("roc_auc_score_train",roc_auc_score_train)
    mlflow.log_artifact("column_list.pkl")
    print(f'Roc Auc Score train: {roc_auc_score_train}  \\n')
    mlflow.log_metric("roc_auc_score_test",roc_auc_score_test)
    print(f'Roc Auc Score test: {roc_auc_score_test}  \\n')
    mlflow.log_metric("training_accuracy",training_accuracy)
    print(f'Accuracy train : {training_accuracy}')
    mlflow.log_metric("test_accuracy",test_accuracy)
    print(f'Accuracy test : {test_accuracy}')
"""),
    md("## 9. 가장 좋은 run을 찾아 모델 레지스트리에 등록 (책 리스팅 4.7)"),
    code("""
from mlflow import MlflowClient

mlflow_client = MlflowClient()
experiment_name = "income-classifier"
experiment = mlflow_client.get_experiment_by_name(experiment_name)
run_object = mlflow_client.search_runs(experiment_ids=experiment.experiment_id,filter_string="metrics.roc_auc_score_test > 0.8",max_results=1,order_by=["metrics.roc_auc_score_test DESC"])[0]
model_uri = f"runs:/{run_object.info.run_id}/{experiment_name}"
registered = mlflow.register_model(model_uri, "random-forest-classifier")
print(run_object.info.run_name, run_object.data.metrics["roc_auc_score_test"], registered.name, registered.version)
"""),
    md("""
[환경] 5장의 추론 컴포넌트는 `model_stage` 값과 같은 stage의 버전을 찾습니다. 책은 stage를 바꾸는 절차를 따로 적지 않았으므로,
여기서 방금 등록한 버전을 `Production`으로 지정합니다. MLflow 3에서 stage는 deprecated 기능이지만 아직 동작합니다.
"""),
    code("""
mlflow_client.transition_model_version_stage("random-forest-classifier", registered.version, "Production")
for v in mlflow_client.search_model_versions("name='random-forest-classifier'"):
    print(v.version, v.current_stage, v.run_id)
"""),
]

if __name__ == "__main__":
    (HERE / "01_mlflow_tracking.ipynb").write_text(
        json.dumps(notebook(mlflow_cells), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print("written", HERE / "01_mlflow_tracking.ipynb")
