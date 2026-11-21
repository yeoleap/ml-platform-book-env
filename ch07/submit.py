"""Uploads a compiled pipeline to KFP and starts one run in the ch07 experiment."""
import argparse
import kfp

p = argparse.ArgumentParser()
p.add_argument("yaml")
p.add_argument("name")
p.add_argument("--run", required=True)
a = p.parse_args()

client = kfp.Client(host="http://127.0.0.1:8080")
pid = client.get_pipeline_id(a.name)
if pid is None:
    pipeline = client.upload_pipeline(a.yaml, pipeline_name=a.name)
    pid = pipeline.pipeline_id
    version_id = client.list_pipeline_versions(pid).pipeline_versions[0].pipeline_version_id
else:
    version_id = client.upload_pipeline_version(a.yaml, pipeline_version_name=a.run, pipeline_id=pid).pipeline_version_id
exp = client.create_experiment("ch07-data-prep")
run = client.run_pipeline(experiment_id=exp.experiment_id, job_name=a.run, pipeline_id=pid, version_id=version_id)
print("run", run.run_id)
print(f"http://127.0.0.1:8080/#/runs/details/{run.run_id}")
