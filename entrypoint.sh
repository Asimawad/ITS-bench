#!/bin/bash
set -e                          
set -o pipefail               

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Entrypoint script started."

export TORCH_LOGS="+dynamo"
export VLLM_TRACE_LEVEL=DEBUG

# --- STEP 1: Start vLLM server using SYSTEM Python ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting vLLM server using system python..."
touch /home/vllm_server.log

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching vLLM server with model: ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"
/usr/bin/python3.11 -m vllm.entrypoints.openai.api_server \
    --model "ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2" \
    --port 8000 \
    --dtype float16 \
    --device cuda \
    --enforce-eager \
    --gpu-memory-utilization 0.9 \
    --max-model-len 13310 \
    --quantization gptq &> /home/vllm_server.log &

while [ ! -f /home/vllm_server.log ]; do 
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for vLLM server log file to be created..."
    sleep 0.2
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting to tail vLLM server logs..."
tail -n +1 -f /home/vllm_server.log &

VLLM_PID=$!
echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM server started with PID: $VLLM_PID, logging to /home/vllm_server.log"

# --- STEP 2: Wait for vLLM server to be healthy ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for vLLM server on port 8000..."
timeout_seconds=1200
start_time=$(date +%s)
while true; do
    current_time=$(date +%s)
    elapsed=$((current_time - start_time))
    if [ $elapsed -ge $timeout_seconds ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ERROR: vLLM server did not become healthy within $timeout_seconds seconds."
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Last 100 lines of vLLM server log:"
        tail -n 100 /home/vllm_server.log
        exit 1
    fi
    
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM server is healthy after $elapsed seconds."
        break
    fi

    echo "[$(date '+%Y-%m-%d %H:%M:%S')] vLLM server is not healthy yet, elapsed time: $elapsed seconds..."
    sleep 1
done

# --- STEP 3: Activate the virtual environment ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Activating virtual environment..."
source /home/.aide-ds/bin/activate
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Virtual environment activated."
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Current PATH: $PATH"

# --- STEP 4: Execute the command passed by Aichor ---
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Executing command: $@"
exec "$@"

