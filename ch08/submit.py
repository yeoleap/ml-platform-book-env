"""Uploads ch08_pipeline.yaml (or a new version) and starts one run in the ch08-id-card experiment."""
import argparse
import kfp

p = argparse.ArgumentParser()
p.add_argument("--run", default="ch08-yolo-e5-img320")
p.add_argument("--epochs", type=int, default=5)
p.add_argument("--imgsz", type=int, default=320)
a = p.parse_args()

name = "ch08-yolo-train-validate"
client = kfp.Client(host="http://127.0.0.1:8080")
pid = client.get_pipeline_id(name)
if pid is None:
    pipeline = client.upload_pipeline("ch08_pipeline.yaml", pipeline_name=name)
    pid = pipeline.pipeline_id
    version_id = client.list_pipeline_versions(pid).pipeline_versions[0].pipeline_version_id
else:
    version_id = client.upload_pipeline_version("ch08_pipeline.yaml", pipeline_version_name=a.run,
                                                pipeline_id=pid).pipeline_version_id
exp = client.create_experiment("ch08-id-card")
run = client.run_pipeline(experiment_id=exp.experiment_id, job_name=a.run, pipeline_id=pid, version_id=version_id,
                          params={"epochs": a.epochs, "imgsz": a.imgsz})
print("run", run.run_id)
print(f"http://127.0.0.1:8080/#/runs/details/{run.run_id}")
