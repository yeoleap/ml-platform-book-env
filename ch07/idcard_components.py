from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Output

S3_ENDPOINT = "http://minio-service.kubeflow:9000"


@dsl.component(base_image="python:3.11", packages_to_install=["boto3"])
def download_dataset_from_s3(
    output_dataset: Output[Dataset],
    source_bucket: str = "ch07-raw",
    source_key: str = "synthetic-idcards.tar.gz",
    output_dir: str = "DATASET",
):
    """Replaces the author's Box download, which now returns 404."""
    import os
    import tarfile

    import boto3

    s3 = boto3.client("s3", endpoint_url="http://minio-service.kubeflow:9000",
                      aws_access_key_id="minio", aws_secret_access_key="minio123")
    s3.download_file(source_bucket, source_key, "DATASET.gz")
    extraction_path = os.path.join(output_dataset.path, output_dir)
    os.makedirs(extraction_path, exist_ok=True)
    with tarfile.open("DATASET.gz", "r:gz") as tar:
        tar.extractall(extraction_path, filter="data")
    for root, _, files in os.walk(extraction_path):
        if files:
            print(root, len(files), "files")


@dsl.component(base_image="python:3.11")
def check_split_pairs(
    train_dataset: Input[Dataset],
    validation_dataset: Input[Dataset],
    test_dataset: Input[Dataset],
    report: Output[Metrics],
    fail_on_error: bool = True,
):
    """Checks every split: image/label pairing by stem, YOLO format, clip leakage."""
    import os

    def labels_in(path):
        return {os.path.splitext(f)[0]: os.path.join(path, f) for f in os.listdir(path)}

    errors, clips = [], {}
    for name, ds in [("train", train_dataset), ("validation", validation_dataset), ("test", test_dataset)]:
        imgs = labels_in(os.path.join(ds.path, "images"))
        lbls = labels_in(os.path.join(ds.path, "labels"))
        unmatched = sorted(set(imgs) ^ set(lbls))
        bad = 0
        for stem, path in lbls.items():
            for line in open(path).read().splitlines():
                parts = line.split()
                if len(parts) != 5 or parts[0] != "0" or not all(0 <= float(v) <= 1 for v in parts[1:]):
                    bad += 1
        clips[name] = {s.split("_")[0] for s in imgs}
        print(f"{name}: images={len(imgs)} labels={len(lbls)} unmatched={len(unmatched)} bad_label_lines={bad}")
        if unmatched:
            print("  e.g.", unmatched[:4])
        report.log_metric(f"{name}_images", len(imgs))
        report.log_metric(f"{name}_unmatched", len(unmatched))
        if unmatched or bad:
            errors.append(f"{name}: {len(unmatched)} unmatched, {bad} bad lines")
    for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
        shared = sorted(clips[a] & clips[b])
        print(f"clips shared by {a} and {b}: {len(shared)} {shared[:6]}")
        report.log_metric(f"shared_clips_{a}_{b}", len(shared))
        if shared:
            errors.append(f"{a}/{b} share clips {shared[:6]}")
    if errors and fail_on_error:
        raise SystemExit("QA failed: " + "; ".join(errors))
    print("QA passed" if not errors else "QA found problems (not failing)")


@dsl.component(base_image="python:3.11")
def split_dataset_by_stem(
    random_state: int,
    input_dataset: Input[Dataset],
    train_dataset: Output[Dataset],
    validation_dataset: Output[Dataset],
    test_dataset: Output[Dataset],
    base_path: str = "MINIDATA",
):
    """Pairs files by stem before splitting and keeps each clip in one split."""
    import os
    import random
    import shutil

    root = os.path.join(input_dataset.path, "DATASET", base_path)
    imgs = {os.path.splitext(f)[0]: f for f in os.listdir(os.path.join(root, "images"))}
    lbls = {os.path.splitext(f)[0]: f for f in os.listdir(os.path.join(root, "labels"))}
    missing = sorted(set(imgs) ^ set(lbls))
    if missing:
        raise SystemExit(f"image/label stems do not match: {missing[:10]}")

    clips = sorted({s.split("_")[0] for s in imgs})
    random.Random(random_state).shuffle(clips)
    n_train = round(len(clips) * 0.75)
    n_val = round(len(clips) * 0.15)
    parts = [clips[:n_train], clips[n_train:n_train + n_val], clips[n_train + n_val:]]

    for out, chosen in zip([train_dataset, validation_dataset, test_dataset], parts):
        os.makedirs(os.path.join(out.path, "images"), exist_ok=True)
        os.makedirs(os.path.join(out.path, "labels"), exist_ok=True)
        stems = sorted(s for s in imgs if s.split("_")[0] in chosen)
        for s in stems:
            shutil.copy2(os.path.join(root, "images", imgs[s]), os.path.join(out.path, "images", imgs[s]))
            shutil.copy2(os.path.join(root, "labels", lbls[s]), os.path.join(out.path, "labels", lbls[s]))
        print(f"{out.uri}: {len(stems)} pairs from clips {sorted(chosen)}")
