"""Author's serving/service.py (BentoML 1.1 Runner API) with the ch10 fixes.

[ch10 fix] /invocation returned only results[0] (one box) and raised IndexError (500) on images
with no card. Return every box, [] when there is none.
[ch10 fix] /render ran with save=True, writing a predictN folder per request, and re-read the file.
Render in memory instead.
[ch10] Every response carries the registry model version, read from model_info.json.
"""
import io
import json
import os

import bentoml
import PIL.Image
from bentoml.exceptions import InvalidArgument
from bentoml.io import JSON, Image
from ultralytics import YOLO

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_INFO = json.load(open(os.path.join(HERE, "model_info.json")))
MODEL_VERSION = f"{MODEL_INFO['model_name']}/{MODEL_INFO['model_version']}"
MAX_PIXELS = 4096 * 4096


class YOLOv8Runnable(bentoml.Runnable):
    SUPPORTED_RESOURCES = ("cpu",)
    SUPPORTS_CPU_MULTI_THREADING = True

    def __init__(self):
        self.model = YOLO(os.path.join(HERE, "yolov8_custom.pt"))
        self.model.cpu()

    @bentoml.Runnable.method(batchable=False)
    def inference(self, input_img):
        result = self.model(input_img.convert("RGB"), imgsz=320, verbose=False)[0]
        return json.loads(result.tojson())

    @bentoml.Runnable.method(batchable=False)
    def render(self, input_img):
        result = self.model(input_img.convert("RGB"), imgsz=320, verbose=False)[0]
        # plot() returns a BGR array; encode to JPEG so the Image output knows the format
        buf = io.BytesIO()
        PIL.Image.fromarray(result.plot()[..., ::-1]).save(buf, format="JPEG")
        buf.seek(0)
        return PIL.Image.open(buf)


yolo_v8_runner = bentoml.Runner(YOLOv8Runnable, name="yolov8_runnable")

svc = bentoml.Service("yolo_v8", runners=[yolo_v8_runner])


def check_size(input_img):
    # keep the PIL image as decoded: the runner serializes it with its own format (JPEG/PNG)
    if input_img.width * input_img.height > MAX_PIXELS:
        raise InvalidArgument(f"image too large: {input_img.width}x{input_img.height}")
    return input_img


@svc.api(input=Image(), output=JSON())
async def invocation(input_img, ctx: bentoml.Context):
    ret = await yolo_v8_runner.inference.async_run(check_size(input_img))
    ctx.response.headers["x-model-version"] = MODEL_VERSION
    return ret


@svc.api(input=Image(), output=Image(mime_type="image/jpeg"))
async def render(input_img, ctx: bentoml.Context):
    ret = await yolo_v8_runner.render.async_run(check_size(input_img))
    ctx.response.headers["x-model-version"] = MODEL_VERSION
    return ret
