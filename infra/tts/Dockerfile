# SGLang 0.5.21 on CUDA 13: the exact base sglang-omni 0.1.8 is built against (and its own Dockerfile pins).
# Skipped from that Dockerfile: the UCX source build, only needed for cross-node (Mooncake) transfers;
# stages on one GPU talk over CUDA IPC and shared memory.
FROM lmsysorg/sglang:v0.5.21@sha256:b1259f3ea3275f66237c498ea388919729018bc9f01c3d638391e06e2cf3f469

# Fish's DAC codec pins protobuf<3.20; keep SGLang's protobuf, as the project's override does.
RUN echo "protobuf>=6.31.1,<7.0.0" > /tmp/overrides.txt \
    && uv pip install --python /opt/sglang/bin/python3 --overrides /tmp/overrides.txt \
        sglang-omni==0.1.8 descript-audiotools==0.7.2 descript-audio-codec==1.0.0

ENV FLASHINFER_WORKSPACE_BASE=/cache
COPY s2pro.yaml /app/s2pro.yaml
