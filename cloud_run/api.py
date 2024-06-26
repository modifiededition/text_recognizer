"""CLOUD RUN function serving text_recognizer predictions."""

import json
import sys
sys.path.append("../")
sys.path.append("../text_recognizer")

from PIL import ImageStat

from text_recognizer.paragraph_text_recognizer import ParagraphTextRecognizer
from text_recognizer import util

from fastapi import FastAPI
#from pydantic import BaseModel
from typing import Dict

model = ParagraphTextRecognizer()
app = FastAPI()

@app.post("/encode/")
async def get_preditiction(raw_image: Dict[str,str]):
    print("INFO loading image")
    image = _load_image(raw_image)
    print(type(image))
    if image is None:
        return {"statusCode": 400, "message": "neither image_url nor image found in event"}

    print("INFO image loaded")
    print("INFO starting inference")
    pred = model.predict(image)
    print("INFO inference complete")
    image_stat = ImageStat.Stat(image)
    print("METRIC image_mean_intensity {}".format(image_stat.mean[0]))
    print("METRIC image_area {}".format(image.size[0] * image.size[1]))
    print("METRIC pred_length {}".format(len(pred)))
    print("INFO pred {}".format(pred))
    return {"pred": str(pred)}


def _load_image(event):
    event = _from_string(event)
    event = _from_string(event.get("body", event))
    image_url = event.get("image_url")
    if image_url is not None:
        print("INFO url {}".format(image_url))
        return util.read_image_pil(image_url, grayscale=True)
    else:
        image = event.get("image")
        if image is not None:
            print("INFO reading image from event")
            return util.read_b64_image(image, grayscale=True)
        else:
            return None


def _from_string(event):
    if isinstance(event, str):
        return json.loads(event)
    else:
        return event
