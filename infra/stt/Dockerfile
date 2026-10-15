# PyTorch 2.14.1 on CUDA 13.0, same CUDA as the vLLM image (host driver >= 580)
FROM pytorch/pytorch:2.14.1-cuda13.0-cudnn9-runtime@sha256:3c0b174a1339193a5bcc4b37d737c9f5ca3c4782a2293d3888e4271367eddf44
WORKDIR /app
# torch lives in the image's system Python; installing next to it is fine inside a container
ENV PIP_BREAK_SYSTEM_PACKAGES=1
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
