FROM python:3.10-slim

WORKDIR /workspace/fmri-fnirs-transfer

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install gdown
RUN pip install gdown

COPY . .

# Download checkpoint during image build
RUN make download_ckpt
