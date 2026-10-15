FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Bake the turn detector's ONNX model into the image, then never reach Hugging Face at runtime.
RUN python -m livekit.agents download-files
ENV HF_HUB_OFFLINE=1
COPY src/ .
CMD ["python", "-m", "livekit.agents", "start", "agent.py"]
