FROM nvcr.io/nvidia/pytorch:25.01-py3

WORKDIR /workspace
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONPATH=/workspace/src
