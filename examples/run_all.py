"""Run steps 1-4 in order. Requires `python bookenv.py forward` in another terminal."""

from pathlib import Path
import subprocess
import sys

here = Path(__file__).resolve().parent
for step in ("01_prepare_data.py", "02_register_features.py", "03_train.py", "04_batch_inference.py"):
    print(f"\n=== {step}", flush=True)
    result = subprocess.run([sys.executable, step], cwd=here)
    if result.returncode:
        sys.exit(f"{step} failed with exit code {result.returncode}")
print("\nAll steps passed.")
