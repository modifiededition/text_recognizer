"""Utilities for model development scripts: training and staging."""
import argparse
import importlib
from io import BytesIO
import base64
from PIL import Image
import smart_open
from typing import Union
from pathlib import Path

DATA_CLASS_MODULE = "text_recognizer.data"
MODEL_CLASS_MODULE = "text_recognizer.models"


def import_class(module_and_class_name: str) -> type:
    """Import class from a module, e.g. 'text_recognizer.models.MLP'."""
    module_name, class_name = module_and_class_name.rsplit(".", 1)
    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def setup_data_and_model_from_args(args: argparse.Namespace):
    data_class = import_class(f"{DATA_CLASS_MODULE}.{args.data_class}")
    model_class = import_class(f"{MODEL_CLASS_MODULE}.{args.model_class}")
    data = data_class(args)
    model = model_class(data_config=data.config(), args=args)
    return data, model


def encode_b64_image(image, format="png"):
    """Encode a PIL image as a base64 string."""
    _buffer = BytesIO()  # bytes that live in memory
    image.save(_buffer, format=format)  # but which we write to like a file
    encoded_image = base64.b64encode(_buffer.getvalue()).decode("utf8")
    return encoded_image


def read_image_pil(image_uri: Union[Path, str], grayscale=False) -> Image:
    with smart_open.open(image_uri, "rb") as image_file:
        return read_image_pil_file(image_file, grayscale)


def read_image_pil_file(image_file, grayscale=False) -> Image:
    with Image.open(image_file) as image:
        if grayscale:
            image = image.convert(mode="L")
        else:
            image = image.convert(mode=image.mode)
        return image



