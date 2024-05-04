"""
The CIFAR-10 dataset
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class.
There are 50000 training images and 10000 test images.
"""

from text_recognizer.metadata.shared import DOWNLOADED_DATA_DIRNAME

INPUT_HEIGHT,INPUT_WIDTH = 32,32
INPUT_CHANNELS = 3
INPUT_SHAPE = (INPUT_CHANNELS, INPUT_WIDTH,INPUT_HEIGHT)
NUM_OF_LABELS = 10

OUTPUT_DIM = (NUM_OF_LABELS,1)

MAPPING = ['plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck']

