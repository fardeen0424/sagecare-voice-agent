# SGLang-Omni runtime (SGLang 0.5.21, CUDA 13, UCX, DAC codec) prebuilt by the project.
# Its only published tag (dev) moves with main, so it is pinned by digest.
FROM hongccc/sglang-omni:dev@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df

# The image clones sglang-omni main on every start; install a pinned release instead.
ENV SGLANG_OMNI_AUTO_CLONE=0
RUN uv pip install --python /opt/sglang/bin/python3 --no-deps sglang-omni==0.1.8

COPY s2pro.yaml /app/s2pro.yaml
