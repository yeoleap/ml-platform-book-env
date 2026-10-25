"""Prepare the chapter 5 book code for this environment.

The book's README asks you to replace `varunmallya` image names with your own
Docker images and to put your Redis password into feature_store.yaml. This
script does both in upstream/chapter-5 so the files stay the book's files.

    python book/ch05/prepare.py
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
CH5 = ROOT / "upstream" / "chapter-5"
IMAGE = "book-ch5/components:0.1"

if not CH5.exists():
    raise SystemExit("upstream/chapter-5 is missing. Run the 'download book code' step first.")

for spec in sorted((CH5 / "components").glob("*/component.yaml")):
    text = spec.read_text(encoding="utf-8")
    new = re.sub(r"image: '[^']+'", f"image: '{IMAGE}'", text)
    spec.write_text(new, encoding="utf-8")
    print(f"{spec.relative_to(ROOT)}: image -> {IMAGE}")

store = CH5 / "scripts" / "feast" / "feature_store.yaml"
text = store.read_text(encoding="utf-8")
text = re.sub(r"password=[^\"]*", "password=redis123", text)
store.write_text(text, encoding="utf-8")
print(f"{store.relative_to(ROOT)}: Redis password set")
