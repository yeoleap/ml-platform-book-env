"""Synthetic ID-card detection data in the layout the author's split code expects.

No real documents or personal data: each image is a drawn card with the word
SAMPLE on a noisy background. Frames of the same clip share a background and
drift slightly, imitating adjacent video frames in MIDV-500.
"""
import argparse
import random
import tarfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

W, H = 320, 240


def make_frame(rng, clip_bg, card, jitter):
    img = Image.new("RGB", (W, H), clip_bg)
    draw = ImageDraw.Draw(img)
    for _ in range(40):
        x, y = rng.randrange(W), rng.randrange(H)
        c = tuple(max(0, min(255, v + rng.randint(-40, 40))) for v in clip_bg)
        draw.ellipse([x, y, x + rng.randint(5, 30), y + rng.randint(5, 30)], fill=c)
    cx, cy, cw, ch, color = card
    cx += rng.randint(-jitter, jitter)
    cy += rng.randint(-jitter, jitter)
    x0, y0 = max(0, cx - cw // 2), max(0, cy - ch // 2)
    x1, y1 = min(W - 1, cx + cw // 2), min(H - 1, cy + ch // 2)
    draw.rounded_rectangle([x0, y0, x1, y1], radius=8, fill=color, outline=(40, 40, 40), width=2)
    draw.rectangle([x0 + 8, y0 + 8, x0 + 8 + (x1 - x0) // 4, y1 - 8], fill=(200, 200, 210))
    draw.text((x0 + (x1 - x0) // 3, y0 + 10), "SAMPLE ID", fill=(20, 20, 60))
    for i in range(3):
        ly = y0 + 30 + i * 14
        if ly < y1 - 6:
            draw.line([x0 + (x1 - x0) // 3, ly, x1 - 10, ly], fill=(90, 90, 90), width=2)
    img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0, 0.8)))
    xc, yc = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
    return img, f"0 {xc:.6f} {yc:.6f} {(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="synthetic-idcards")
    p.add_argument("--clips", type=int, default=12)
    p.add_argument("--frames", type=int, default=10)
    p.add_argument("--seed", type=int, default=7)
    a = p.parse_args()
    rng = random.Random(a.seed)
    root = Path(a.out) / "MINIDATA"
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for c in range(1, a.clips + 1):
        bg = tuple(rng.randint(60, 200) for _ in range(3))
        card = (rng.randint(110, 210), rng.randint(90, 150), rng.randint(140, 200),
                rng.randint(90, 120), tuple(rng.randint(200, 250) for _ in range(3)))
        for f in range(1, a.frames + 1):
            stem = f"clip{c:02d}_frame{f:03d}"
            img, label = make_frame(rng, bg, card, jitter=6)
            img.save(root / "images" / f"{stem}.png")
            (root / "labels" / f"{stem}.txt").write_text(label)
    archive = Path(a.out).with_suffix(".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(root, arcname="MINIDATA")
    print(f"{a.clips * a.frames} images, archive {archive} ({archive.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
