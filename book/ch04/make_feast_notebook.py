"""Build the chapter 4 Feast notebook from the book's code (practical-mlops/chapter-4)."""

import json
from pathlib import Path

from make_notebooks import code, md, notebook

HERE = Path(__file__).resolve().parent

cells = [
    md("""
# 4.2 Feast를 피처 스토어로 사용하기

책 4.2절과 공식 저장소 `chapter-4/Readme.md`의 Feast 절차를 순서대로 옮겼습니다.
책의 명령은 터미널용이지만, 여기서는 같은 명령을 셀에서 실행합니다. `[환경]` 표시 셀만 이 실습 환경을 위해 추가했습니다.

실행 전 확인할 것
1. 다른 터미널에서 `python bookenv.py forward`가 실행 중이어야 합니다. Feast는 S3(9000)와 Redis(6379) 두 터널을 모두 사용합니다.
2. 저장소 UI에서 `feature-data-sets`, `feature-registry` 버킷을 만들어 두어야 합니다.
"""),
    md("[환경] 경로와 명령 실행 도우미"),
    code("""
from pathlib import Path
import shutil, subprocess, sys

UPSTREAM = Path("../../upstream/chapter-4").resolve()
REPO = Path("feature_repo").resolve()
FEAST = str(Path(sys.executable).parent / "feast")

def run(*args, cwd=REPO):
    print("$", " ".join(args))
    result = subprocess.run([FEAST if args[0] == "feast" else args[0], *args[1:]],
                            cwd=cwd, capture_output=True, text=True, encoding="utf-8")
    print(result.stdout[-3000:])
    if result.returncode:
        print(result.stderr[-3000:])
        raise RuntimeError(f"exit code {result.returncode}")

assert (UPSTREAM / "feast").exists(), "SOP의 '책 코드 내려받기' 단계를 먼저 실행하세요."
"""),
    md("## 1. 환경 변수 (README의 export 4줄)\n\nFeast와 boto3가 S3 대신 MinIO 주소로 접속하도록 지정합니다."),
    code("""
import os
os.environ["AWS_SECRET_ACCESS_KEY"] = "minio123"
os.environ["AWS_ACCESS_KEY_ID"] = "minio"
os.environ["FEAST_S3_ENDPOINT_URL"] = "http://localhost:9000"
os.environ["AWS_ENDPOINT_URL"] = "http://localhost:9000"
"""),
    md("## 2. 피처 원천 파일을 `feature-data-sets` 버킷으로 옮기기\n\nREADME의 \"move the datasets in feast/feature_data to minio bucket feature-data-sets\" 단계입니다. 저장소 UI의 Upload로 올려도 결과는 같습니다."),
    code("""
from minio import Minio
client = Minio("localhost:9000", access_key="minio", secret_key="minio123", secure=False)
assert client.bucket_exists("feature-data-sets"), "feature-data-sets 버킷을 먼저 만드세요."
for path in sorted((UPSTREAM / "feast" / "feature_data").glob("*.parquet")):
    client.fput_object("feature-data-sets", path.name, str(path))
    print("uploaded", path.name)
print([o.object_name for o in client.list_objects("feature-data-sets")])
"""),
    md("## 3. 피처 저장소 폴더 준비 (책 리스팅 4.8~4.11)\n\n`entity.py`, `features.py`, `feature_store.yaml`을 그대로 복사합니다. README 안내대로 `feature_store.yaml`의 Redis 비밀번호만 바꿉니다."),
    code("""
import re
REPO.mkdir(exist_ok=True)
for name in ("entity.py", "features.py", "feature_store.yaml"):
    shutil.copy(UPSTREAM / "feast" / name, REPO / name)
config = (REPO / "feature_store.yaml").read_text(encoding="utf-8")
config = re.sub(r"password=[^\\"]*", "password=redis123", config)
(REPO / "feature_store.yaml").write_text(config, encoding="utf-8")
print(config)
"""),
    code("""
print((REPO / "features.py").read_text(encoding="utf-8")[:900])
"""),
    md("## 4. `feast apply` (책 4.2.1)\n\n정의를 `s3://feature-registry/registry_local.db`에 기록하고, Redis 온라인 저장소를 준비합니다. 값은 아직 Redis에 들어가지 않습니다."),
    code("""
run("feast", "apply")
"""),
    md("## 5. 과거 시점 피처 조회 (책 4.2.2, `feast_retrieval/retrieve_features.py`)"),
    code("""
shutil.copy(UPSTREAM / "feast_retrieval" / "retrieve_features.py", REPO / "retrieve_features.py")
run(sys.executable, "retrieve_features.py")
"""),
    md("## 6. 온라인 저장소로 적재 (`feast materialize`)\n\nREADME의 기간을 그대로 사용합니다."),
    code("""
run("feast", "materialize", "2022-09-16T00:00:00", "2023-09-17T00:00:00")
"""),
    md("## 7. 온라인 피처 조회 (`feast_retrieval/retrieve_online_features.py`)"),
    code("""
shutil.copy(UPSTREAM / "feast_retrieval" / "retrieve_online_features.py", REPO / "retrieve_online_features.py")
run(sys.executable, "retrieve_online_features.py")
"""),
    md("## 8. `feature_store.yaml`을 `feature-registry` 버킷에 올리기\n\nREADME 마지막 단계입니다. 5장 파이프라인은 이 위치에서 설정 파일을 내려받습니다."),
    code("""
run(sys.executable, str(UPSTREAM / "scripts" / "push_feature_store_config_to_minio.py"),
    "--bucket-name=feature-registry", "--minio-endpoint=localhost:9000",
    "--access-key=minio", "--secret-key=minio123", "--insecure",
    "--feature-store-path", str(REPO / "feature_store.yaml"))
print([o.object_name for o in client.list_objects("feature-registry")])
"""),
]

(HERE / "02_feast_feature_store.ipynb").write_text(
    json.dumps(notebook(cells), ensure_ascii=False, indent=1), encoding="utf-8"
)
print("written", HERE / "02_feast_feature_store.ipynb")
