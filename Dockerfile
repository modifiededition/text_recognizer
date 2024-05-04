FROM python:3.10-slim

# Install Python dependencies
COPY requirements/prod.txt ./requirements.txt
RUN pip install --upgrade pip==23.1.2
RUN pip install --no-cache-dir -r requirements.txt

# Copy only the relevant directories and files
#   note that we use a .dockerignore file to avoid copying logs etc.
COPY text_recognizer/ ./text_recognizer

COPY ./cloud_run ./
EXPOSE 8080
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8080"]