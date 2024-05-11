
# Handwritten Text Recognizer

This repository contains the code base for a handwritten text recognizer. The machine learning model architecture is based on the image-to-sequence paradigm. For image encoding, 
a ResNet architecture is employed, while decoding utilizes a Transformer architecture. The model comprises 14 million parameters and has been trained over 20 hours using a Tesla T4 machine.
Training data is sourced from the IAM paragraph dataset, augmented with synthetic data to improve performance.
Through data augmentation, the validation Character Error Rate (CER) has been reduced from 0.78 to 0.23.

# Stack
The following technologies have been employed to build the model and the product:

- PyTorch: Utilized for constructing the model architecture.
- PyTorch Lightning: Employed for training and tracking model metrics.
- Weights and Biases: Used for experiment tracking and model versioning. Training Report: https://api.wandb.ai/links/ashishgy77/tfyktbdu
- TorchScript: Utilized for model packaging, enabling integration into production systems.
- FASTAPI: Employed to build a REST API for model deployment, with the frontend built using Gradio.
- Docker: The API and server have been containerized and deployed as a microservice on Google Cloud Run.
- Evidently: Integrated for monitoring pipeline, enabling real-time tracking of drift in image statistics and text descriptors.
- GitHub Actions: Employed for continuous integration, including linting, unit testing, and continuous deployment to build Docker images and submit to the repository.

# DEMO

[LINK]


