#!/bin/bash
set -e

echo "=== RunPod2 vLLM Startup ==="
echo "Model: ${MODEL_NAME}"
echo "K_DIM: ${K_DIM}"
echo "SEQ_LEN: ${SEQ_LEN}"

# Detect number of GPUs
GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l || echo "0")
echo "Detected GPUs: ${GPU_COUNT}"

if [ "$GPU_COUNT" -eq "0" ]; then
    echo "ERROR: No GPUs detected"
    exit 1
fi

# Use actual GPU count (auto-detect). MiniMax-M2.7 prod profile expects TP=2 (2xH200) or TP=4 (4xH100/H200).
TP_SIZE=${GPU_COUNT}
echo "Tensor Parallel Size: ${TP_SIZE}"

# vLLM server settings
VLLM_PORT=${VLLM_PORT:-8000}
VLLM_HOST=${VLLM_HOST:-127.0.0.1}

echo ""
echo "=== Starting vLLM Server ==="
echo "Port: ${VLLM_PORT}"
echo "Host: ${VLLM_HOST}"

# Start vLLM server in background with MiniMax-M2.7 flags matching kaitakuai's prod profile
# and modal/test1.py:291-308. Do NOT change without updating both.
/usr/bin/python3.12 -m vllm.entrypoints.openai.api_server \
    --model "${MODEL_NAME}" \
    --host "${VLLM_HOST}" \
    --port "${VLLM_PORT}" \
    --trust-remote-code \
    --tensor-parallel-size "${TP_SIZE}" \
    --attention-backend FLASHINFER \
    --moe-backend triton \
    --gpu-memory-utilization 0.92 \
    --max-num-seqs 128 \
    --enable-auto-tool-choice \
    --max-model-len 180000 \
    --kv-cache-dtype fp8 \
    --logprobs-mode processed_logprobs \
    --tool-call-parser minimax_m2 \
    --reasoning-parser minimax_m2_append_think \
    2>&1 | tee /tmp/vllm.log &

VLLM_PID=$!
echo "vLLM started with PID: ${VLLM_PID}"

# Start RunPod handler immediately (handler.py will wait for vLLM health
# while polling orchestrator for shutdown commands, so the worker can be
# stopped even during model loading)
echo ""
echo "=== Starting RunPod Handler ==="
echo "Handler will wait for vLLM readiness while polling orchestrator"
exec /usr/bin/python3.12 /app/handler.py
