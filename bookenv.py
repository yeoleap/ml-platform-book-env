"""Set up the "Machine Learning Platform Engineering" chapter 4-5 environment on a local kind cluster.

Usage:
    python bookenv.py check     # verify docker, kind, kubectl, git
    python bookenv.py up        # create the cluster and install everything
    python bookenv.py forward   # port-forward S3, storage UI, MLflow, Redis, KFP UI (Ctrl+C to stop)
    python bookenv.py buckets   # create every bucket the book uses (shortcut for the manual step)
    python bookenv.py load IMG  # load a locally built image into the cluster
    python bookenv.py status    # show pods per namespace
    python bookenv.py down      # delete the cluster
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
CLUSTER = "book-env"
CONTEXT = f"kind-{CLUSTER}"
KFP_VERSION = "2.4.1"
KFP_CACHE = ROOT / ".cache" / f"kfp-{KFP_VERSION}"
KFP_MANIFESTS = KFP_CACHE / "manifests" / "kustomize"
IMAGES = {
    "book-env/mlflow:3.0.0": ROOT / "images" / "mlflow" / "Dockerfile",
    "book-env/runtime:0.1": ROOT / "images" / "runtime" / "Dockerfile",
}
FORWARDS = [
    ("S3 API (MinIO 호환)", "kubeflow", "svc/minio-service", "9000:9000"),
    ("저장소 웹 UI (SeaweedFS Filer)", "kubeflow", "deploy/minio", "8888:8888"),
    ("MLflow", "mlflow", "svc/mlflow-service", "5000:5000"),
    ("Redis", "redis", "svc/redis-deployment-master", "6379:6379"),
    ("KFP UI", "kubeflow", "svc/ml-pipeline-ui", "8080:80"),
]


def tool(name):
    override = os.environ.get(name.upper())
    if override:
        return override
    found = shutil.which(name)
    if not found:
        sys.exit(f"'{name}' was not found on PATH. See README for installation.")
    return found


def run(*args, check=True, capture=False, retries=1, delay=10):
    for attempt in range(1, retries + 1):
        print("$", " ".join(str(a) for a in args), flush=True)
        result = subprocess.run(
            [str(a) for a in args], text=True, encoding="utf-8", errors="replace",
            capture_output=capture,
        )
        if result.returncode == 0 or attempt == retries:
            break
        print(f"  retrying in {delay}s ({attempt}/{retries})", flush=True)
        time.sleep(delay)
    if check and result.returncode != 0:
        if capture:
            print(result.stdout, result.stderr)
        sys.exit(f"Command failed with exit code {result.returncode}")
    return result


def kubectl(*args, **kwargs):
    return run(tool("kubectl"), "--context", CONTEXT, *args, **kwargs)


def check():
    for name in ("docker", "kind", "kubectl", "git"):
        print(f"{name:8s} {tool(name)}")
    run(tool("docker"), "info", "--format", "Docker {{.ServerVersion}}")
    print("All required tools are available.")


def cluster_exists():
    out = run(tool("kind"), "get", "clusters", capture=True).stdout.split()
    return CLUSTER in out


def fetch_kfp_manifests():
    if (KFP_MANIFESTS / "env" / "platform-agnostic").is_dir():
        return
    if KFP_CACHE.exists():
        shutil.rmtree(KFP_CACHE)
    git = tool("git")
    run(git, "clone", "--quiet", "--depth", "1", "--branch", KFP_VERSION,
        "--filter=blob:none", "--sparse",
        "https://github.com/kubeflow/pipelines.git", KFP_CACHE, retries=3)
    run(git, "-C", KFP_CACHE, "sparse-checkout", "set", "manifests/kustomize", retries=3)


def install_kfp():
    fetch_kfp_manifests()
    kubectl("apply", "-k", KFP_MANIFESTS / "cluster-scoped-resources")
    kubectl("wait", "--for", "condition=established", "--timeout=120s",
            "crd/applications.app.k8s.io")
    kubectl("apply", "-k", KFP_MANIFESTS / "env" / "platform-agnostic", retries=3)
    kubectl("-n", "kubeflow", "patch", "deployment", "minio",
            "--patch-file", ROOT / "manifests" / "kfp" / "object-store-patch.yaml")
    kubectl("-n", "kubeflow", "patch", "deployment", "workflow-controller",
            "--patch-file", ROOT / "manifests" / "kfp" / "executor-patch.yaml")


def build_images():
    docker, kind = tool("docker"), tool("kind")
    for tag, dockerfile in IMAGES.items():
        run(docker, "build", "-t", tag, "-f", dockerfile, ROOT, retries=2)
        run(kind, "load", "docker-image", tag, "--name", CLUSTER)


def rollout(namespace, *deployments, timeout="900s"):
    for name in deployments:
        kubectl("-n", namespace, "rollout", "status", f"deployment/{name}",
                f"--timeout={timeout}")


def up():
    check()
    if cluster_exists():
        print(f"Cluster '{CLUSTER}' already exists; reusing it.")
    else:
        run(tool("kind"), "create", "cluster", "--config",
            ROOT / "cluster" / "kind.yaml", "--wait", "180s")
    install_kfp()
    services = ROOT / "manifests" / "services"
    kubectl("apply", "-f", services / "postgres.yaml", "-f", services / "redis.yaml")
    build_images()
    kubectl("apply", "-f", services / "mlflow.yaml")
    rollout("kubeflow", "minio", "mysql", "metadata-grpc-deployment",
            "ml-pipeline", "workflow-controller", "ml-pipeline-ui")
    rollout("postgres", "postgres-release-postgresql")
    rollout("redis", "redis-deployment-master")
    rollout("mlflow", "mlflow-deployment")
    print("\nThe environment is ready. Next: python bookenv.py forward")


def buckets():
    services = ROOT / "manifests" / "services"
    kubectl("-n", "kubeflow", "delete", "job", "create-buckets",
            "--ignore-not-found=true")
    kubectl("apply", "-f", services / "buckets.yaml")
    kubectl("-n", "kubeflow", "wait", "--for=condition=complete",
            "--timeout=300s", "job/create-buckets")
    kubectl("-n", "kubeflow", "logs", "job/create-buckets")


def load(image):
    run(tool("kind"), "load", "docker-image", image, "--name", CLUSTER)


def start_forward(namespace, target, ports):
    return subprocess.Popen([
        tool("kubectl"), "--context", CONTEXT, "-n", namespace,
        "port-forward", "--address", "127.0.0.1", target, ports,
    ], stdout=subprocess.DEVNULL)


def forward():
    # A tunnel ends when its Pod restarts, so dropped tunnels are reopened.
    processes = {}
    try:
        for label, namespace, target, ports in FORWARDS:
            print(f"{label:28s} http://localhost:{ports.split(':')[0]}")
            processes[(namespace, target, ports)] = start_forward(namespace, target, ports)
        print("\nPort-forwarding is active. Press Ctrl+C to stop.", flush=True)
        while True:
            time.sleep(2)
            for key, process in processes.items():
                if process.poll() is not None:
                    print(f"Reconnecting {key[1]}", flush=True)
                    time.sleep(2)
                    processes[key] = start_forward(*key)
    except KeyboardInterrupt:
        pass
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            process.wait()


def status():
    for namespace in ("kubeflow", "mlflow", "postgres", "redis"):
        print(f"\n== {namespace}")
        kubectl("-n", namespace, "get", "pods", check=False)


def down():
    if cluster_exists():
        run(tool("kind"), "delete", "cluster", "--name", CLUSTER)
    else:
        print(f"Cluster '{CLUSTER}' does not exist.")


def main():
    commands = {"check": check, "up": up, "forward": forward, "buckets": buckets,
                "status": status, "down": down}
    if len(sys.argv) == 3 and sys.argv[1] == "load":
        load(sys.argv[2])
        return
    if len(sys.argv) != 2 or sys.argv[1] not in commands:
        sys.exit(__doc__)
    commands[sys.argv[1]]()


if __name__ == "__main__":
    main()
