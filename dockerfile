FROM nvcr.io/nvidia/pytorch:24.10-py3

RUN apt-get update && \
    apt-get install -y espeak-ng && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /zonos

COPY . /zonos

RUN pip install uv

RUN uv venv && \
    uv sync --no-group main && \
    uv sync
