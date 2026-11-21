"""8.1.7 manual review. Saves plotted images instead of cv2.imshow (no GUI needed)."""
import glob, os, sys
import cv2
from ultralytics import YOLO

weights = sys.argv[1] if len(sys.argv) > 1 else "yolov8n.pt"
out = sys.argv[2] if len(sys.argv) > 2 else "review_out"
os.makedirs(out, exist_ok=True)
model = YOLO(weights)
for file in sorted(glob.glob("samples/*.jpg")):
    result = model(cv2.imread(file), imgsz=320, verbose=False)
    r = result[0]
    names = [f"{r.names[int(c)]}:{float(s):.2f}" for c, s in zip(r.boxes.cls, r.boxes.conf)]
    print(os.path.basename(file), names if names else "no detection")
    cv2.imwrite(os.path.join(out, os.path.basename(file)), r.plot())
