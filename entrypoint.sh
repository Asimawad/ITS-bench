#!/bin/bash

# Print commands and their arguments as they are executed
set -x

{
  # log into /home/logs
  LOGS_DIR=./logs
  mkdir -p $LOGS_DIR

  # chmod the /home directory such that nonroot users can work on everything within it. We do this at container start
  # time so that anything added later in agent-specific Dockerfiles will also receive the correct permissions.
  # (this command does `chmod a+rw /home` but with the exception of /home/data, which is a read-only volume)
  find /home -path /home/data -prune -o -exec chmod a+rw {} \;
  # ls -l .

  # Launch grading server, stays alive throughout container lifetime to service agent requests.
  /opt/conda/bin/python /private/grading_server.py
} 2>&1 | tee $LOGS_DIR/entrypoint.log
# RedHatAI/DeepSeek-R1-Distill-Qwen-32B-FP8-dynamic
# "ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2"
# Start vLLM server in the background
echo "Starting vLLM server..."
python -m vllm.entrypoints.openai.api_server \
    --model "ModelCloud/DeepSeek-R1-Distill-Qwen-7B-gptqmodel-4bit-vortex-v2" \
    --port 8000 \
    --dtype bfloat16 \
    --device cuda \
    --gpu-memory-utilization 0.9 \
    --max-model-len 13310 \
    --quantization gptq  &
VLLM_PID=$!
Wait
echo "vLLM server started with PID: $VLLM_PID"

# Wait for servers to initialize
echo "Waiting for servers to initialize..."
sleep 50

# Debug: Print current working directory and list files
echo "Current working directory: $(pwd)"
echo "Files in current directory:"
ls -la

# Debug: Print the command to be executed
echo "Executing command: $@"

# Execute the command passed to the container
exec "$@"