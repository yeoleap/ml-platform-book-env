"""Synthetic ID-card-like detection dataset in YOLO format. No real person or document data."""
import argparse
import os
import random
import tarfile

from PIL import Image, ImageDraw, ImageFilter, ImageFont

W, H = 480, 360


def background(rng):
    img = Image.new("RGB", (W, H), tuple(rng.randint(40, 220) for _ in range(3)))
    d = ImageDraw.Draw(img)
    for _ in range(rng.randint(8, 25)):
        x0, y0 = rng.randint(-50, W), rng.randint(-50, H)
        x1, y1 = x0 + rng.randint(20, 220), y0 + rng.randint(20, 160)
        color = tuple(rng.randint(0, 255) for _ in range(3))
        if rng.random() < 0.5:
            d.rectangle([x0, y0, x1, y1], fill=color)
        else:
            d.ellipse([x0, y0, x1, y1], fill=color)
    return img.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 2.5)))


def card(rng, cw):
    ch = int(cw / 1.586)
    base = tuple(rng.randint(170, 250) for _ in range(3))
    c = Image.new("RGBA", (cw, ch), (0, 0, 0, 0))
    d = ImageDraw.Draw(c)
    d.rounded_rectangle([0, 0, cw - 1, ch - 1], radius=max(4, cw // 20), fill=base + (255,))
    header = tuple(rng.randint(20, 140) for _ in range(3))
    d.rectangle([0, int(ch * 0.05), cw, int(ch * 0.22)], fill=header + (255,))
    font = ImageFont.load_default()
    d.text((int(cw * 0.05), int(ch * 0.08)), "SYNTHETIC ID", fill=(255, 255, 255, 255), font=font)
    # photo placeholder (abstract silhouette, not a face image)
    px0, py0 = int(cw * 0.06), int(ch * 0.30)
    px1, py1 = int(cw * 0.32), int(ch * 0.90)
    d.rectangle([px0, py0, px1, py1], fill=(150, 160, 175, 255))
    cx = (px0 + px1) // 2
    r = (px1 - px0) // 4
    d.ellipse([cx - r, py0 + r // 2, cx + r, py0 + r // 2 + 2 * r], fill=(90, 95, 110, 255))
    d.rectangle([px0 + r // 2, py1 - (py1 - py0) // 3, px1 - r // 2, py1], fill=(90, 95, 110, 255))
    for i in range(5):
        y = int(ch * (0.34 + i * 0.11))
        x1 = int(cw * rng.uniform(0.6, 0.95))
        d.rectangle([int(cw * 0.38), y, x1, y + max(2, ch // 30)], fill=(60, 60, 60, 255))
    return c


def place(bg, cimg, cx, cy, angle):
    rot = cimg.rotate(angle, expand=True, resample=Image.BICUBIC)
    x0, y0 = int(cx - rot.width / 2), int(cy - rot.height / 2)
    bg.paste(rot, (x0, y0), rot)
    bbox = rot.getchannel("A").getbbox()  # axis-aligned box of the visible card pixels
    bx0, by0, bx1, by1 = bbox[0] + x0, bbox[1] + y0, bbox[2] + x0, bbox[3] + y0
    return max(0, bx0), max(0, by0), min(W, bx1), min(H, by1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="synthetic")
    ap.add_argument("--clips", type=int, default=30)
    ap.add_argument("--frames", type=int, default=6)
    ap.add_argument("--negatives", type=int, default=12)
    ap.add_argument("--seed", type=int, default=8)
    ap.add_argument("--tar", default="DATASET.tar.gz")
    a = ap.parse_args()
    rng = random.Random(a.seed)
    img_dir = os.path.join(a.out, "MINIDATA", "images")
    lbl_dir = os.path.join(a.out, "MINIDATA", "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    n = 0
    for clip in range(a.clips):
        bg0 = background(rng)
        cimg = card(rng, rng.randint(140, 300))
        cx, cy = rng.uniform(0.3, 0.7) * W, rng.uniform(0.3, 0.7) * H
        ang = rng.uniform(-25, 25)
        for f in range(a.frames):
            bg = bg0.copy()
            box = place(bg, cimg, cx + rng.uniform(-12, 12), cy + rng.uniform(-10, 10), ang + rng.uniform(-4, 4))
            if rng.random() < 0.3:
                bg = bg.filter(ImageFilter.GaussianBlur(rng.uniform(0.5, 1.5)))
            stem = f"clip{clip:02d}_f{f:02d}"
            bg.save(os.path.join(img_dir, stem + ".jpg"), quality=88)
            bx0, by0, bx1, by1 = box
            with open(os.path.join(lbl_dir, stem + ".txt"), "w") as fh:
                fh.write(f"0 {(bx0 + bx1) / 2 / W:.6f} {(by0 + by1) / 2 / H:.6f} {(bx1 - bx0) / W:.6f} {(by1 - by0) / H:.6f}\n")
            n += 1
    for i in range(a.negatives):
        stem = f"neg{i:02d}"
        background(rng).save(os.path.join(img_dir, stem + ".jpg"), quality=88)
        open(os.path.join(lbl_dir, stem + ".txt"), "w").close()
        n += 1
    with tarfile.open(a.tar, "w:gz") as tar:
        tar.add(os.path.join(a.out, "MINIDATA"), arcname="MINIDATA")
    print(f"images={n} clips={a.clips} frames_per_clip={a.frames} negatives={a.negatives} tar={a.tar} size={os.path.getsize(a.tar)}")


if __name__ == "__main__":
    main()
