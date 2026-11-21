"""Author's training_and_validation_pipeline.py with the ch08 fixes, adapted for an arm64 Mac.

[mac] train and validate both run on ultralytics/ultralytics:8.3.40-arm64
(torch 2.5.1 CPU, torchvision 0.20.1, numpy 1.23.5, libGL included), so the author's
apt-get / pip install step and the amd64 8.0.194-cpu image (emulated on arm64) are gone.
"""
from kfp import dsl
from kfp.dsl import Input, Output, Dataset, Model, Artifact, Metrics

YOLO_IMAGE = "ultralytics/ultralytics:8.3.40-arm64"


@dsl.component(
    packages_to_install=["boto3"],
    base_image="python:3.11"
)
def download_dataset(output_dataset: Output[Dataset], output_dir: str = "DATASET",
                     s3_endpoint: str = "http://minio-service.kubeflow.svc.cluster.local:9000",
                     bucket: str = "ch08-datasets", key: str = "DATASET.tar.gz"):
    # [ch08 fix] The author's Manning box URL now returns 404.
    # Read the same tar.gz layout (MINIDATA/images, MINIDATA/labels) from S3 instead.
    import os
    import tarfile
    import boto3

    downloaded_file = "DATASET.gz"
    s3 = boto3.client("s3", endpoint_url=s3_endpoint,
                      aws_access_key_id="minio", aws_secret_access_key="minio123")
    s3.download_file(bucket, key, downloaded_file)
    print(f"downloaded s3://{bucket}/{key} ({os.path.getsize(downloaded_file)} bytes)")

    # Extract to the output directory
    extraction_path = os.path.join(output_dataset.path, output_dir)
    os.makedirs(extraction_path, exist_ok=True)

    with tarfile.open(downloaded_file, 'r:gz') as tar:
        tar.extractall(extraction_path, filter="data")


@dsl.component(base_image="python:3.11")
def split_dataset(
        random_state: int,
        input_dataset: Input[Dataset],
        train_dataset: Output[Dataset],
        validation_dataset: Output[Dataset],
        test_dataset: Output[Dataset]
):
    import os
    import glob
    import random
    import shutil

    BASE_PATH = "MINIDATA"

    raw_images = list(glob.glob(os.path.join(input_dataset.path, "DATASET", BASE_PATH, "images", "**")))
    raw_labels = list(glob.glob(os.path.join(input_dataset.path, "DATASET", BASE_PATH, "labels", "**")))

    # [ch08 fix] The author zips two independently globbed lists. glob order is not
    # guaranteed, so image i and label i may belong to different samples.
    stem = lambda p: os.path.splitext(os.path.basename(p))[0]
    mismatched = sum(stem(a) != stem(b) for a, b in zip(raw_images, raw_labels))
    print(f"author-style zip(glob, glob): {mismatched}/{len(raw_images)} pairs have different stems")
    label_by_stem = {stem(p): p for p in raw_labels}
    images = sorted(p for p in raw_images if stem(p) in label_by_stem)
    labels = [label_by_stem[stem(p)] for p in images]
    print(f"paired by stem: {len(images)} samples, images without label: {len(raw_images) - len(images)}")

    train_ratio = 0.75
    validation_ratio = 0.15

    # [ch08 fix] The author splits frames at random, so frames of one clip land in
    # train and validation. Split whole clips instead (each negNN image is its own group).
    clip = lambda p: stem(p).split("_")[0]
    clips = sorted({clip(p) for p in images})
    random.Random(random_state).shuffle(clips)
    n_train = round(len(clips) * train_ratio)
    n_val = round(len(clips) * validation_ratio)
    split_of = {c: 0 for c in clips[:n_train]}
    split_of.update({c: 1 for c in clips[n_train:n_train + n_val]})
    split_of.update({c: 2 for c in clips[n_train + n_val:]})
    parts = [[], [], []]
    for x, y in zip(images, labels):
        parts[split_of[clip(x)]].append((x, y))

    # Create output directories for each split
    for dataset_output, pairs in zip([train_dataset, validation_dataset, test_dataset], parts):
        os.makedirs(os.path.join(dataset_output.path, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_output.path, "labels"), exist_ok=True)
        for x, y in pairs:
            shutil.copy2(x, os.path.join(dataset_output.path, "images", os.path.basename(x)))
            shutil.copy2(y, os.path.join(dataset_output.path, "labels", os.path.basename(y)))

    # [ch08 check] adjacent-frame leakage: frames of one clip landing in several splits
    train_clips, val_clips, test_clips = ({clip(x) for x, _ in p} for p in parts)
    print(f"split sizes train={len(parts[0])} val={len(parts[1])} test={len(parts[2])}")
    print(f"val clips also in train: {len(val_clips & train_clips)}/{len(val_clips)}")
    print(f"test clips also in train: {len(test_clips & train_clips)}/{len(test_clips)}")
    print(f"val clips: {sorted(val_clips)}")
    print(f"test clips: {sorted(test_clips)}")


TRAINING_SCRIPT = '''
import os
import yaml
import shutil
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--train-path', required=True, help='Path to training dataset')
    parser.add_argument('--val-path', required=True, help='Path to validation dataset')
    parser.add_argument('--test-path', required=True, help='Path to test dataset')
    parser.add_argument('--epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--batch', type=int, required=True, help='Batch size')
    parser.add_argument('--model-name', required=True, help='Name of the model')
    parser.add_argument('--model-output', required=True, help='Path to save the model')
    parser.add_argument('--data-yaml', required=True, help='Path to save data.yaml')
    parser.add_argument('--imgsz', type=int, default=640, help='[ch08] image size (CPU lab uses 320)')
    return parser.parse_args()

def main():
    args = parse_args()

    # Define data configuration using absolute paths
    data = {
        'train': os.path.join(args.train_path, "images"),
        'val': os.path.join(args.val_path, "images"),
        'test': os.path.join(args.test_path, "images"),
        'nc': 1,
        'names': {
            0: 'id_card'
        }
    }

    # Write data.yaml
    data_yaml_path = os.path.join(args.data_yaml, "data.yaml")
    os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)

    print(f"Writing data configuration to {data_yaml_path}")
    print("Data YAML contents:")
    print(yaml.dump(data))

    with open(data_yaml_path, 'w') as file:
        yaml.dump(data, file)

    # Train model
    model = YOLO('yolov8n.pt')
    results = model.train(
        data=data_yaml_path,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=0,
        project=os.path.dirname(args.model_output),
        name=args.model_name
    )

    # Save best model
    best_model_source = os.path.join(os.path.dirname(args.model_output), args.model_name, "weights", "best.pt")
    best_model_dest = os.path.join(args.model_output, "best.pt")
    os.makedirs(os.path.dirname(best_model_dest), exist_ok=True)
    shutil.copy2(best_model_source, best_model_dest)

    # [ch08 fix] project=dirname(model_output) is outside the artifact folder, so plots,
    # results.csv and last.pt were never uploaded. Keep the run folder inside the Model artifact.
    shutil.copytree(os.path.join(os.path.dirname(args.model_output), args.model_name),
                    os.path.join(args.model_output, "train_run"), dirs_exist_ok=True)
    print("model artifact contents:", sorted(os.listdir(args.model_output)))

if __name__ == "__main__":
    main()
'''


@dsl.container_component
def train_model(
        epochs: int,
        batch: int,
        yolo_model_name: str,
        train_dataset: dsl.Input[dsl.Dataset],
        validation_dataset: dsl.Input[dsl.Dataset],
        test_dataset: dsl.Input[dsl.Dataset],
        model_output: dsl.Output[dsl.Model],
        data_yaml: dsl.Output[dsl.Artifact],
        imgsz: int = 640
):
    return dsl.ContainerSpec(
        image=YOLO_IMAGE,
        command=['bash', '-c'],
        args=[
            f'''
            # [mac] ultralytics, torch and libGL are already in the image: no apt-get / pip install

            # Write training script
            cat << 'EOF' > /tmp/train.py
{TRAINING_SCRIPT}
EOF

            # Execute the training script
            python3 /tmp/train.py \
                --train-path "$0" \
                --val-path "$1" \
                --test-path "$2" \
                --epochs "$3" \
                --batch "$4" \
                --model-name "$5" \
                --model-output "$6" \
                --data-yaml "$7" \
                --imgsz "$8"
            ''',
            train_dataset.path,
            validation_dataset.path,
            test_dataset.path,
            epochs,
            batch,
            yolo_model_name,
            model_output.path,
            data_yaml.path,
            imgsz
        ]
    )


@dsl.component(base_image=YOLO_IMAGE)
def validate_model(
        data_yaml: Input[Artifact],
        model: Input[Model],
        validation_dataset: Input[Dataset],
        test_dataset: Input[Dataset],
        metrics: Output[Metrics],
        imgsz: int = 640
):
    from ultralytics import YOLO
    import os

    # Print paths for debugging
    print(f"Data YAML path: {data_yaml.path}")
    print(f"Model path: {model.path}")
    print(f"Validation dataset path: {validation_dataset.path}")
    print(f"Test dataset path: {test_dataset.path}")

    # Load the trained model
    model_path = os.path.join(model.path, "best.pt")
    print(f"Loading model from: {model_path}")
    model = YOLO(model_path)

    # Run validation using the existing data.yaml
    validation_results = model.val(
        data=os.path.join(data_yaml.path, "data.yaml"),
        imgsz=imgsz,
        batch=1,
        verbose=True
    )

    # Log metrics
    metrics.log_metric("map50-95", validation_results.box.map)
    metrics.log_metric("map50", validation_results.box.map50)
    metrics.log_metric("map75", validation_results.box.map75)

    # [ch08 fix] best.pt was chosen on the validation set, so its score is not independent.
    # Also score the test split, which training never looked at.
    test_results = model.val(
        data=os.path.join(data_yaml.path, "data.yaml"),
        split="test",
        imgsz=imgsz,
        batch=1,
        verbose=True
    )
    metrics.log_metric("test_map50-95", test_results.box.map)
    metrics.log_metric("test_map50", test_results.box.map50)


@dsl.pipeline(
    name="ch08-yolo-train-validate",
    description="Chapter 8 YOLO train/validate on a synthetic ID-card dataset (CPU, arm64)"
)
def pipeline(
        epochs: int = 5,
        batch: int = 8,
        random_state: int = 42,
        yolo_model_name: str = "yolov8n_custom",
        imgsz: int = 320,
        dataset_bucket: str = "ch08-datasets"
):
    # Download dataset
    download_op = download_dataset(bucket=dataset_bucket)

    # Split dataset
    split_op = split_dataset(
        random_state=random_state,
        input_dataset=download_op.outputs["output_dataset"]
    )

    # Train model using split data
    train_op = train_model(
        epochs=epochs,
        batch=batch,
        yolo_model_name=yolo_model_name,
        train_dataset=split_op.outputs["train_dataset"],
        validation_dataset=split_op.outputs["validation_dataset"],
        test_dataset=split_op.outputs["test_dataset"],
        imgsz=imgsz
    )
    # [mac] limit alone also sets request=limit; the 8GB kind node has ~3Gi unreserved,
    # so reserve little and let the pod burst up to the limit.
    train_op.set_cpu_request("1").set_cpu_limit("4").set_memory_request("1G").set_memory_limit("3G")

    # Validate model
    validate_op = validate_model(
        data_yaml=train_op.outputs["data_yaml"],
        model=train_op.outputs["model_output"],
        validation_dataset=split_op.outputs["validation_dataset"],
        test_dataset=split_op.outputs["test_dataset"],
        imgsz=imgsz
    )
    validate_op.set_cpu_request("1").set_cpu_limit("2").set_memory_request("1G").set_memory_limit("2G")


if __name__ == '__main__':
    from kfp import compiler
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path='ch08_pipeline.yaml'
    )
