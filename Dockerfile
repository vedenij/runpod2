# kaitakuai's mlnode image tuned for MiniMax-M2.7 on H200 (vLLM 0.20.0 + PoC v2 patches)
FROM ghcr.io/kaitakuai/mlnode-h200-minimax-m2-7:0.2.13-vllm0.20.0-k1

WORKDIR /app

# Ensure `python` / `python3` resolve to the image's Python 3.12
RUN ln -sf /usr/bin/python3.12 /usr/local/bin/python3 && ln -sf /usr/bin/python3.12 /usr/local/bin/python

# Install RunPod SDK and HTTP client
RUN python -m pip install --no-cache-dir runpod requests httpx

# Copy handler code
COPY handler.py /app/
COPY startup.sh /app/
RUN chmod +x /app/startup.sh

# Environment variables
ENV PYTHONUNBUFFERED=1

# PoC v2 / model settings
ENV MODEL_NAME=MiniMaxAI/MiniMax-M2.7
ENV K_DIM=12
ENV SEQ_LEN=1024

# vLLM settings
ENV VLLM_PORT=8000
ENV VLLM_HOST=127.0.0.1
ENV VLLM_RPC_TIMEOUT=120000
ENV VLLM_USE_V1=1
ENV VLLM_USE_FLASHINFER_MOE_FP8=0
ENV VLLM_ALLOW_INSECURE_SERIALIZATION=1
ENV WATCHER_GRACE_FIRST_HEALTHY=1

# NCCL settings for multi-GPU communication
ENV NCCL_NVLS_ENABLE=0
ENV NCCL_P2P_DISABLE=1
ENV NCCL_IB_DISABLE=1
ENV NCCL_SHM_DISABLE=0
ENV NCCL_DEBUG=WARN

# HuggingFace cache location (RunPod network volume with pre-downloaded weights)
ENV HF_HOME=/runpod-volume/huggingface-cache
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub
ENV HF_XET_HIGH_PERFORMANCE=1

# Clear any default entrypoint from the base image
ENTRYPOINT []

# Run startup script which starts vLLM and handler
CMD ["/app/startup.sh"]
