#!/bin/bash
set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Exit if any command in a pipeline fails.

echo "Entrypoint script started."

# --- STEP 1: Start vLLM server using SYSTEM Python ---
echo "Starting vLLM server using system python..."
# Redirect stdout and stderr to a log file
/usr/bin/python3.11 -m vllm.entrypoints.openai.api_server \
    --model "ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2" \
    --port 8000 \
    --dtype bfloat16 \
    --device cuda \
    --gpu-memory-utilization 0.9 \
    --max-model-len 13310 \
    --quantization gptq &> /home/vllm_server.log &

VLLM_PID=$!
echo "vLLM server started with PID: $VLLM_PID, logging to /home/vllm_server.log"

# --- STEP 2: Wait for vLLM server to be healthy ---
echo "Waiting for vLLM server on port 8000..."
timeout_seconds=1200 # Give it some time to start
start_time=$(date +%s)
while true; do
    current_time=$(date +%s)
    if [ $(($current_time - $start_time)) -ge $timeout_seconds ]; then
        echo "vLLM server did not become healthy within $timeout_seconds seconds."
        exit 1
    fi
    if curl -s http://localhost:8000/health > /dev/null; then
        echo "vLLM server is healthy."
        break
    fi
    echo "vLLM server is not healthy yet, time passed -> $(($current_time - $start_time)) ."

    sleep 1
done

# --- STEP 3: Activate the virtual environment ---
echo "Activating virtual environment..."
# Ensure this path is correct based on your Dockerfile install location
source /home/.aide-ds/bin/activate
echo "Virtual environment activated."
echo "Current PATH: $PATH" # Verify PATH includes venv bin first

# --- STEP 4: Execute the command passed by Aichor ---
# This command should be your run.sh or a direct call to run_agent.py
echo "Executing command: $@"
exec "$@"

