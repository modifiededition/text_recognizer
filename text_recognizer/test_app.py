import json
import os

import requests
import torch

from handwritting_text_recognizer import app
from handwritting_text_recognizer import util
from handwritting_text_recognizer.text_recognizer.metadata import iam_paragraphs as metadata
from handwritting_text_recognizer.text_recognizer.models import ResnetTransformer

os.environ["CUDA_VISIBLE_DEVICES"] = ""

def test_front_end_and_backend():
    """A quick test to make sure we can build the app and ping the API locally."""
    backend = app.PredictorBackend()
    frontend = app.make_frontend(fn=backend.run)
    # run the UI without blocking
    frontend.launch(share=False, prevent_thread_lock=True)
    local_url = frontend.local_url
    get_response = requests.get(local_url)
    assert get_response.status_code == 200, get_response.content
    image_b64 = util.encode_b64_image(util.read_image_pil("a01-000u.png"))
    local_api = f"{local_url}api/predict"
    headers = {"Content-Type": "application/json"}
    payload = json.dumps({"data": ["data:image/png;base64," + image_b64]})
    post_response = requests.post(local_api, data=payload, headers=headers)
    assert post_response.status_code == 200, post_response.content


def test_model():
    
    input_dims = metadata.DIMS
    mapping = metadata.MAPPING
    output_dims = metadata.OUTPUT_DIMS

    data_config = {
        "input_dims" : input_dims,
        "mapping" : mapping,
        "output_dims" : output_dims,
        }
    model  = ResnetTransformer(data_config=data_config)

    # input tensor
    input_tensor = torch.zeros(1, 1, 100, 100)
    output_tensor = model(input_tensor)
    assert output_tensor.shape == (1,682)