"""Stress cases for manual review: transformations the training data never showed."""
import os
from PIL import Image, ImageFilter
import cv2
from ultralytics import YOLO
os.makedirs("stress", exist_ok=True)
src = Image.open("samples/clip01_f00.jpg")
cases = {
    "rot90": src.rotate(90, expand=True),
    "blur8": src.filter(ImageFilter.GaussianBlur(8)),
    "small": Image.new("RGB", src.size, (128, 128, 128)),
    "dark": src.point(lambda v: int(v * 0.25)),
}
cases["small"].paste(src.resize((src.width // 4, src.height // 4)), (200, 150))
m = YOLO("kfp_artifacts/model_output/best.pt")
for k, im in cases.items():
    p = f"stress/{k}.jpg"; im.save(p)
    r = m(cv2.imread(p), imgsz=320, verbose=False)[0]
    print(k, [f"{float(s):.2f}" for s in r.boxes.conf] or "no detection")
    cv2.imwrite(f"stress/{k}_pred.jpg", r.plot())
